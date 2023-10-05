import numpy as np
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, AveragePooling2D, Dropout, BatchNormalization,SpatialDropout2D
from keras.utils import to_categorical
from keras.layers import LeakyReLU
from keras.layers import ReLU

from tensorflow.keras.applications import ResNet101V2, ResNet50V2, Xception, InceptionV3, DenseNet201, MobileNetV2, VGG16

## GLOBAL SEED ##    
np.random.seed(42)
tf.random.set_seed(42)

def build_vanilla_cnn(ks,ps,type_pooling,stc,stp,do,md,nfilters,activation):
    num_classes = 5
    if activation == 'LeakyReLU':
        activation_conv= LeakyReLU()
    elif activation == 'ReLU':
        activation_conv= ReLU()
        
    padding_type = 'same'
    model = Sequential()
    
    model.add(Conv2D(nfilters, kernel_size=(ks, ks),activation=activation_conv,
        input_shape=(240, 720, 1),padding=padding_type,strides=stc))
    
    if type_pooling == 'max':
        model.add(MaxPooling2D((ps, ps),padding=padding_type,strides=stp))
    elif type_pooling == 'avg':
        model.add(AveragePooling2D((ps, ps),padding=padding_type,strides=stp))
        
    model.add(Dropout(do))
    model.add(Conv2D(nfilters*2, (ks, ks), activation=activation_conv,padding=padding_type,strides=stc))
    
    if type_pooling == 'max':
        model.add(MaxPooling2D((ps, ps),padding=padding_type,strides=stp))
    elif type_pooling == 'avg':
        model.add(AveragePooling2D((ps, ps),padding=padding_type,strides=stp))
        
    model.add(SpatialDropout2D(do))

    model.add(Flatten())
    model.add(Dense(num_classes*md*md, activation=activation_conv))
    model.add(Dense(num_classes*md, activation=activation_conv))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def build_predesigned_model(name_model,type_pooling,do,md,activation,input_shape):
    num_classes = 5
    if activation == 'LeakyReLU':
        activation_conv= LeakyReLU()
    elif activation == 'ReLU':
        activation_conv= ReLU()
        
    model = Sequential()
    if name_model=='resnet50':
        model.add(ResNet50V2(weights=None, include_top=False, input_shape=input_shape,pooling=type_pooling))
    if name_model=='resnet101':
        model.add(ResNet101V2(weights=None, include_top=False, input_shape=input_shape,pooling=type_pooling))
    if name_model=='xception':
        model.add(Xception(weights=None, include_top=False, input_shape=input_shape,pooling=type_pooling))
    if name_model=='inception':
        model.add(InceptionV3(weights=None, include_top=False, input_shape=input_shape,pooling=type_pooling))
    if name_model=='densenet':
        model.add(DenseNet201(weights=None, include_top=False, input_shape=input_shape,pooling=type_pooling))
    if name_model=='mobilenet':
        model.add(MobileNetV2(weights=None, include_top=False, input_shape=input_shape,pooling=type_pooling))
    if name_model=='vgg16':
        model.add(VGG16(weights=None, include_top=False, input_shape=input_shape,pooling=type_pooling))
    if name_model=='vgg19':
        model.add(VGG19(weights=None, include_top=False, input_shape=input_shape,pooling=type_pooling))
    
    model.add(Dropout(do))

    model.add(Flatten())

    model.add(Dense(num_classes*md*md, activation=activation_conv))
    model.add(Dense(num_classes*md, activation=activation_conv))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


