from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add, Cropping2D, \
    ZeroPadding2D, Dropout, UpSampling2D
from keras.models import Model
from customLayers import InstanceNormalization


def conv2D_BN_Act(nFeature, kSize, kStride, inp, padding='same', isInstNorm=False):
    C = Conv2D(nFeature, kernel_size=kSize, strides=kStride, padding=padding)(inp)
    if isInstNorm:
        C_BN = InstanceNormalization()(C)
    else:
        C_BN = BatchNormalization()(C)
    C_BN_DO = Dropout(0.2)(C_BN)
    C_BN_Act = Activation('relu')(C_BN_DO)
    return C_BN_Act


def conv2DTrans_BN_Act(nFeature, kSize, kStride, crop, outPad, concateInp, inp, padding='same'):
    C = Conv2DTranspose(nFeature, kernel_size=kSize, strides=kStride, padding=padding)(inp)
    C_Crop = Cropping2D(cropping=crop)(C)
    C_Zpad = ZeroPadding2D(padding=outPad)(C_Crop)
    C_Con = concatenate([C_Zpad, concateInp], axis=-1)
    return C_Con


def UpSampling2D_BN_Act(kSize, crop, outPad, concateInp, inp, padding='same'):
    C = UpSampling2D(size=kSize)(inp)
    C_Crop = Cropping2D(cropping=crop)(C)
    C_Zpad = ZeroPadding2D(padding=outPad)(C_Crop)
    C_Con = concatenate([C_Zpad, concateInp], axis=-1)
    return C_Con


def inceptionResNetV2_enc_dec_1(H=256, W=256, weights='imagenet', noutchannel=1, isregression=True, ismuticlass=False,
                                isRes=False):
    """
    This is an encoder-decoder network architecture for image reconstruction and image segmentation
    based on pretrained inceptionResNetV2 model as encoder. The deocder need to be trained

    :param img_sz: size of the input image [n x n x 3]
    :param noutchannel: The number of output channels of the network, on;y valid if ismuticlass==True else ignored
    :param isregression: If ==True then model output is single channel without any activation applied to the last layer
                         If ==False and ismuticlass==False, then output is single channel and sigmoid activation applied to the last layer
                         If ==False and ismuticlass==True, then output is multichannel determined by noutchannel and softmax activation is appled to the last layer
    :param ismuticlass:  If ==False then output is single channel and sigmoid activation applied to the last layer
                         If ==True then output is multichannel determined by noutchannel and softmax activation is appled to the last layer
    :return: Encoder Decoder Model
    """
    model = InceptionResNetV2(include_top=False, weights=weights, input_tensor=None, input_shape=(H, W, 3),
                              pooling=None)

    # ----- Decoder Network ------ #
    nFeat_d = 256
    convT_0_0_d = conv2DTrans_BN_Act(nFeature=nFeat_d, kSize=2, kStride=2, crop=0, outPad=1,
                                     concateInp=model.get_layer('activation_162').output,
                                     inp=model.get_layer('conv_7b').output)
    convT_0_0_d = concatenate([convT_0_0_d, model.get_layer('activation_159').output,
                               model.get_layer('activation_157').output], axis=-1)
    conv_0_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_0_0_d)
    conv_0_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_0_0_d)

    convT_1_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=((0, 1), (0, 1)),
                                      concateInp=model.get_layer('activation_75').output, inp=conv_0_1_d)
    conv_1_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_1_0_d)
    conv_1_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_1_0_d)

    convT_2_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=1,
                                      concateInp=model.get_layer('activation_5').output, inp=conv_1_1_d)
    conv_2_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_2_0_d)
    conv_2_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_2_0_d)

    convT_3_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=((2, 3), (2, 3)),
                                      concateInp=model.get_layer('activation_3').output, inp=conv_2_1_d)
    conv_3_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_3_0_d)
    conv_3_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_3_0_d)

    convT_4_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=3,
                                      concateInp=model.input, inp=conv_3_1_d)
    conv_4_0_d = conv2D_BN_Act(nFeature=int(nFeat_d / 2), kSize=3, kStride=1, inp=convT_4_0_d)
    conv_4_1_d = conv2D_BN_Act(nFeature=int(nFeat_d / 2), kSize=3, kStride=1, inp=conv_4_0_d)

    if (isregression):
        if isRes:
            conv_4_2_d = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(conv_4_1_d)
            conv_4_2_d = add([conv_4_2_d, model.input])
        else:
            conv_4_2_d = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(conv_4_1_d)
    else:
        if (ismuticlass):
            conv_4_2_d = Conv2D(filters=noutchannel, kernel_size=3, strides=1, padding='same')(conv_4_1_d)
            conv_4_2_d = Activation('softmax')(conv_4_2_d)
        else:
            conv_4_2_d = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(conv_4_1_d)
            conv_4_2_d = Activation('sigmoid')(conv_4_2_d)

    model_1 = Model(inputs=[model.input], outputs=[conv_4_2_d])
    return model_1
