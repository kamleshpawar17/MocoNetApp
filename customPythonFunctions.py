import pydicom
import glob
import numpy as np
import os


def readDicomFrmFolder(dcmPath, extension='*.dcm'):
    '''
    This function reads dicom images into a 3D numpy array
    :param dcmPath: Path to dicom folder
    :return:
    Img3D: 3d image as numpy array
    ds: Header for one dcm image
    orient: orientation of first two dimension either 'Sag', 'Tra' or 'Cor'
    '''
    filenames = glob.glob(os.path.join(dcmPath, extension))
    slcs = len(filenames)
    assert (slcs > 0), "No dicom files found in the source folder"
    ds = pydicom.dcmread(filenames[0], force=True)
    rows = int(ds[(0x28, 0x10)].value)
    cols = int(ds[(0x28, 0x11)].value)
    Img3D = np.zeros((rows, cols, slcs))
    for f in filenames:
        ds = pydicom.dcmread(f, force=True)
        slc = int(ds[(0x20, 0x13)].value)
        Img3D[:, :, slc - 1] = ds.pixel_array
    return Img3D, ds


def writeDicom_to_Folder(dicomPathSrc, dicomPathDst, img):
    '''
    This function writes the data from numpy array to dicom files
    :param dicomPathSrc: names of the files from which header should come
    :param dicomPathDst: folder where to write new files
    :param img: numpy array uint16
    :return:
    '''
    filenames = glob.glob(os.path.join(dicomPathSrc, '*.dcm'))
    if not os.path.exists(dicomPathDst):
        os.mkdir(dicomPathDst)
    for f in filenames:
        ds = pydicom.dcmread(f, force=False)
        slc = int(ds[(0x20, 0x13)].value)
        temp = np.squeeze(img[:, :, slc - 1])
        ds.PixelData = temp.tostring()
        fname = 'mc_' + f.split('/')[-1]
        fdst = os.path.join(dicomPathDst, fname)
        ds.save_as(fdst)

