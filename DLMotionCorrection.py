import numpy as np
import progressbar
import nibabel as nb

from customPythonFunctions import readDicomFrmFolder, writeDicom_to_Folder
from models import inceptionResNetV2_enc_dec_1


# ---- Add Noise Class ----- #
class add_noise():
    def fft2(self, x):
        y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm='ortho'))
        return y

    def ifft2(self, x):
        y = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x), norm='ortho'))
        return y

    def gaussian(self, x, frac=0.02):
        mu = np.mean(x)
        sigma = np.std(x)
        xn = 1 / sigma * (x - mu)
        noise = np.random.normal(0, frac, np.shape(x))
        y = np.abs((xn + noise) * sigma + mu).astype('uint16')
        return y

    def rician(self, x, frac=0.02):
        mu = np.mean(x)
        sigma = np.std(x)
        xn = 1 / sigma * (x - mu)
        xn_fft = self.fft2(xn)
        noise = np.random.normal(0, frac, np.shape(x)) + 1j * np.random.normal(0, frac, np.shape(x))
        xn_fft = xn_fft + noise
        yn = self.ifft2(xn_fft)
        y = np.abs(yn * sigma + mu).astype('uint16')
        return y


# --- Image normalization class ---- #
class normalize_meanstd():
    def __init__(self, x, nstd=6.0, sigma_dst=1 / 6., mu_dst=0.0):
        self.mu_src = np.mean(x)
        self.sigma_src = np.std(x)
        self.eps = 1e-12
        self.nstd = nstd
        self.sigma_dst = sigma_dst
        self.mu_dst = mu_dst

    def normalize(self, x):
        y = (x - self.mu_src)
        y = y / (self.eps + self.sigma_src)
        # y[y < -self.nstd] = -self.nstd
        # y[y > self.nstd] = self.nstd
        y = self.sigma_dst * y + self.mu_dst
        return y

    def denormalize(self, x):
        y = (x - self.mu_dst)
        y = y / self.sigma_dst
        y = self.sigma_src * y + self.mu_src
        return y


# --- Image resize class ---- #
class image_zpad():
    def __init__(self):
        self.r = None
        self.c = None

    def zpad_to_32n(self, img):
        self.r, self.c, _ = img.shape
        self.zpad_r = (np.ceil(self.r / 32.) * 32.).astype('float32') - self.r
        self.zpad_c = (np.ceil(self.c / 32.) * 32.).astype('float32') - self.c
        self.zpad_r0 = int(np.ceil(self.zpad_r / 2.))
        self.zpad_r1 = int(np.floor(self.zpad_r / 2.))
        self.zpad_c0 = int(np.ceil(self.zpad_c / 2.))
        self.zpad_c1 = int(np.floor(self.zpad_c / 2.))
        img = np.pad(img, ((self.zpad_r0, self.zpad_r1), (self.zpad_c0, self.zpad_c1), (0, 0)), 'constant')
        return img

    def remove_zpad(self, img):
        img = img[self.zpad_r0:self.zpad_r0 + self.r, self.zpad_c0:self.zpad_c0 + self.c, :]
        return img


class DL_MC():
    def __init__(self, isout3ch=False, isin3ch=True, mu_dst=0.5, sigma_dst=1 / 6.,
                 noise_frac=0.04):
        self.add_noise = add_noise()
        self.isout3ch = isout3ch
        self.isin3ch = isin3ch
        self.mu_dst = mu_dst
        self.sigma_dst = sigma_dst
        self.noise_frac = noise_frac

    def __call__(self, dicomPathSrc, dicomPathDst, *args, **kwargs):
        # ---- Read Dicom into Numpy Array ----- #
        img, ds = readDicomFrmFolder(dicomPathSrc)

        # ---- Change orientation to make first two dimension 'sagital view' ---- #
        smallest_indx = np.argsort(img.shape)[0]
        if smallest_indx == 2:
            img_orient = img.copy()
        elif smallest_indx == 1:
            img_orient = np.transpose(img, (0, 2, 1))
        else:
            img_orient = np.transpose(img, (1, 2, 0))

        # ---- Preprocess Image ---- #
        zpad_obj = image_zpad()
        img_resz_zpad = zpad_obj.zpad_to_32n(img_orient)
        img_norm = normalize_meanstd(img_resz_zpad, mu_dst=self.mu_dst, sigma_dst=self.sigma_dst)
        img_inp = img_norm.normalize(img_resz_zpad)

        # ------ Model Definition -------- #
        H, W, SLC = img_inp.shape
        model = inceptionResNetV2_enc_dec_1(H=H, W=W, weights=None, noutchannel=1, isregression=True, ismuticlass=False,
                                            isRes=False)
        model.load_weights('./weights/model-mc-incpeResV2.hdf5')

        # ---- Predict ---- #
        img_out = np.zeros((H, W, SLC))
        slc_count = 0.0
        bar = progressbar.ProgressBar(maxval=SLC).start()
        for k in range(SLC):
            # print('Completed: ', format(slc_count / SLC * 100., '0.2f'), '%')
            x = np.expand_dims(np.expand_dims(img_inp[:, :, k], -1), 0)
            if self.isin3ch:
                x = np.concatenate((x, x, x), axis=-1)
            y_pred = np.squeeze(model.predict(x))
            if self.isout3ch:
                img_out[:, :, k] = y_pred[:, :, 0]
            else:
                img_out[:, :, k] = y_pred
            slc_count += 1
            bar.update(k + 1)

        # ---- Postprocess Image ---- #
        img_out = img_norm.denormalize(img_out)
        img_out_zpad_removed = zpad_obj.remove_zpad(img_out)

        # ---- re-orient to original view ---- #
        if smallest_indx == 2:
            img_mc = img_out_zpad_removed.copy()
        elif smallest_indx == 1:
            img_mc = np.transpose(img_out_zpad_removed, (0, 2, 1))
        else:
            img_mc = np.transpose(img_out_zpad_removed, (2, 0, 1))

        img_mc = np.abs(img_mc).astype('uint16')
        img_mc = self.add_noise.rician(img_mc, frac=self.noise_frac)
        # ----- write dicom output ---- #
        writeDicom_to_Folder(dicomPathSrc, dicomPathDst, img_mc)
        # --- Write nifit output file ---- #
        imgInpObj = nb.Nifti1Image(img_mc, np.eye(4))
        nb.save(imgInpObj, './static/data/processedFilesNifiti/output.nii')
