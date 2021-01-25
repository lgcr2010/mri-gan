from scipy.fftpack import fft2,fftshift,ifftshift,ifft2
from scipy.io import loadmat, savemat
import numpy as n
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import math
import numpy as n
from tensorflow.keras import layers

path = 'mask/Poisson2D/'
file_name = os.listdir(path)
mask = []
for i in file_name:
    maskfl = loadmat(path+i)
    print(maskfl)
    mask.append(maskfl['population_matrix'])


def to_bad_img(x, mask):
    x = (x + 1.) / 2.
    fft = fft2(x[:, :, 0])
    fft = fftshift(fft)
    fft = fft * mask
    fft = ifftshift(fft)
    x = ifft2(fft)
    x = n.abs(x)
    x = x * 2 - 1
    return x[:, :, n.newaxis]

def fft_abs_for_map_fn(x):
    x = (x + 1.) / 2.
    x_complex = tf.dtypes.complex(x, tf.zeros_like(x))[:, :, 0]
    fft = tf.signal.fft2d(x_complex)
    fft_abs = tf.math.abs(fft)
    return fft_abs

'''
#################################################################################################
        

def gene():
    
    inputs = layers.Input(shape=[256,256,1])
    
    x1 = layers.Conv2D(filters=64,
                       kernel_size=[8,8],
                       strides=(2, 2),
                       use_bias=False,
                       input_shape=(256,256,1))(inputs)#1
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(alpha =0.4)(x1)
    x2 = layers.Conv2D(filters=128,
                       kernel_size=[8,8],
                       use_bias=False)(x1)#2
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(alpha =0.4)(x2)
    x3 = layers.Conv2D(filters=256,
                       kernel_size=[6,6],
                       use_bias=False)(x2)#3
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.LeakyReLU(alpha =0.4)(x3)
    x4 = layers.Conv2D(filters=512,
                       kernel_size=[6,6],
                       use_bias=False)(x3)#4
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.LeakyReLU(alpha =0.4)(x4)
    x5 = layers.Conv2D(filters=512,
                       kernel_size=[6,6],
                       use_bias=False)(x4)#5
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.LeakyReLU(alpha =0.4)(x5)
    x6 = layers.Conv2D(filters=512,
                       kernel_size=[4,4],
                       use_bias=False)(x5)#6
    x6 = layers.BatchNormalization()(x6)
    x6 = layers.LeakyReLU(alpha =0.4)(x6)
    x7 = layers.Conv2D(filters=512,
                       kernel_size=[3,3],
                       use_bias=False)(x6)#7
    x7 = layers.BatchNormalization()(x7)
    x7 = layers.LeakyReLU(alpha =0.4)(x7)
    x8 = layers.Conv2D(filters=512,
                       kernel_size=[3,3],
                       use_bias=False)(x7)#8
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.LeakyReLU(alpha =0.4)(x8)
    y1 = tf.keras.layers.Conv2DTranspose(512,
                                        kernel_size=[3,3],
                                        use_bias=False)(x8)#1
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.LeakyReLU(alpha =0.4)(y1)
    y1 = layers.Add()([x7,y1])
    y2 = tf.keras.layers.Conv2DTranspose(512,
                                        kernel_size=[3,3],
                                        use_bias=False)(y1)#2
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.LeakyReLU(alpha =0.4)(y2)
    y2 = layers.Add()([x6,y2])
    y3 = tf.keras.layers.Conv2DTranspose(512,
                                        kernel_size=[4,4],
                                        use_bias=False)(y2)#3
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.LeakyReLU(alpha =0.4)(y3)
    y3 = layers.Add()([x5,y3])
    y4 = tf.keras.layers.Conv2DTranspose(512,
                                        kernel_size=[6,6],
                                        use_bias=False)(y3)#4
    y4 = layers.BatchNormalization()(y4)
    y4 = layers.LeakyReLU(alpha =0.4)(y4)
    y4 = layers.Add()([x4,y4])
    y5 = tf.keras.layers.Conv2DTranspose(256,
                                        kernel_size=[6,6],
                                        use_bias=False)(y4)#5
    y5 = layers.BatchNormalization()(y5)
    y5 = layers.LeakyReLU(alpha =0.4)(y5)
    y5 = layers.Add()([x3,y5])
    y6 = tf.keras.layers.Conv2DTranspose(128,
                                        kernel_size=[6,6],
                                        use_bias=False)(y5)#6
    y6 = layers.BatchNormalization()(y6)
    y6 = layers.LeakyReLU(alpha =0.4)(y6)
    y6 = layers.Add()([x2,y6])
    y7 = tf.keras.layers.Conv2DTranspose(64,
                                        kernel_size=[8,8],
                                        use_bias=False)(y6)#7
    y7 = layers.BatchNormalization()(y7)
    y7 = layers.LeakyReLU(alpha =0.4)(y7)
    y7 = layers.Add()([x1,y7])
    y8 = tf.keras.layers.Conv2DTranspose(64,
                                        kernel_size=[8,8],
                                        strides=(2, 2),
                                        use_bias=False)(y7)#8
    y8 = layers.BatchNormalization()(y8)
    y8 = layers.LeakyReLU(alpha =0.4)(y8)
    y8 = layers.Add()([x7,y1])
    x = layers.Conv2D(1,
                       kernel_size=[1,1],
                       activation='tanh')(y8)
    outputs = x
    
    model = tf.keras.Model(inputs = inputs,outputs = outputs)
    
    return model


'''

class gene(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=[4,4],
                                            strides=(2, 2),
                                            padding='same',
                                           input_shape=(256,256,1))#output = [128,128,64]
        self.Norm1 = tf.keras.layers.BatchNormalization()
        self.leak1 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.conv2 = tf.keras.layers.Conv2D(filters=128,
                                           kernel_size=[4,4],
                                            strides=(2, 2),
                                            padding='same')#output = [64,64,128]
        self.Norm2 = tf.keras.layers.BatchNormalization()
        self.leak2 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.conv3 = tf.keras.layers.Conv2D(filters=256,
                                           kernel_size=[4,4],
                                            strides=(2, 2),
                                            padding='same')#output = [32,32,256]
        self.Norm3 = tf.keras.layers.BatchNormalization()
        self.leak3 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.conv4 = tf.keras.layers.Conv2D(filters=512,
                                           kernel_size=[4,4],
                                            strides=(2, 2),
                                            padding='same',)#output = [16,16,512]
        self.Norm4 = tf.keras.layers.BatchNormalization()
        self.leak4 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.conv5 = tf.keras.layers.Conv2D(filters=512,
                                           kernel_size=[4,4],
                                            padding='same',
                                            strides=(2, 2))#output = [8,8,512]
        self.Norm5 = tf.keras.layers.BatchNormalization()
        self.leak5 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.conv6 = tf.keras.layers.Conv2D(filters=512,
                                           kernel_size=[4,4],
                                            padding='same',
                                            strides=(2, 2))#output = [4,4,512]
        self.Norm6 = tf.keras.layers.BatchNormalization()
        self.leak6 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.conv7 = tf.keras.layers.Conv2D(filters=512,
                                           kernel_size=[3,3],
                                            padding='same',
                                            strides=(2, 2))#output = [2,2,512]
        self.Norm7 = tf.keras.layers.BatchNormalization()
        self.leak7 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.conv8 = tf.keras.layers.Conv2D(filters=512,
                                           kernel_size=[2,2],
                                            strides=(2,2),
                                            padding='same')#output = [1,1,512]
        self.Norm8 = tf.keras.layers.BatchNormalization()
        self.leak8 = tf.keras.layers.LeakyReLU(alpha =0.4)
        
        self.deconv1 = tf.keras.layers.Conv2DTranspose(512,
                                        kernel_size=[2,2],
                                        strides=(2,2),
                                        padding='same')#output = [2,2,512]
        self.deNorm1 = tf.keras.layers.BatchNormalization()
        self.deleak1 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.add1 = tf.keras.layers.Add()
        self.deconv2 = tf.keras.layers.Conv2DTranspose(512,
                                        kernel_size=[3,3],
                                        padding='same',
                                        strides=(2,2))#output = [4,4,512]
        self.deNorm2 = tf.keras.layers.BatchNormalization()
        self.deleak2 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.add2 = tf.keras.layers.Add()
        self.deconv3 = tf.keras.layers.Conv2DTranspose(512,
                                        kernel_size=[4,4],
                                        strides=(2, 2),
                                        padding='same')#output = [8,8,512]
        self.deNorm3 = tf.keras.layers.BatchNormalization()
        self.deleak3 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.add3 = tf.keras.layers.Add()
        self.deconv4 = tf.keras.layers.Conv2DTranspose(512,
                                        kernel_size=[4,4],
                                        strides=(2, 2),
                                        padding='same')#output = [16,16,512]
        self.deNorm4 = tf.keras.layers.BatchNormalization()
        self.deleak4 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.add4 = tf.keras.layers.Add()
        self.deconv5 = tf.keras.layers.Conv2DTranspose(256,
                                        kernel_size=[4,4],
                                        strides=(2, 2),
                                        padding='same')#output = [32,32,256]
        self.deNorm5 = tf.keras.layers.BatchNormalization()
        self.deleak5 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.add5 = tf.keras.layers.Add()
        self.deconv6 = tf.keras.layers.Conv2DTranspose(128,
                                        kernel_size=[4,4],
                                        strides=(2, 2),
                                        padding='same')#output = [64,64,128]
        self.deNorm6 = tf.keras.layers.BatchNormalization()
        self.deleak6 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.add6 = tf.keras.layers.Add()
        self.deconv7 = tf.keras.layers.Conv2DTranspose(64,
                                        kernel_size=[4,4],
                                        strides=(2, 2),
                                        padding='same')#output = [128,128,64]
        self.deNorm7 = tf.keras.layers.BatchNormalization()
        self.deleak7 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.add7= tf.keras.layers.Add()
        self.deconv8 = tf.keras.layers.Conv2DTranspose(64,
                                        kernel_size=[4,4],
                                        strides=(2, 2),
                                        padding='same')#output = [256,256,64]
        self.deNorm8 = tf.keras.layers.BatchNormalization()
        self.deleak8 = tf.keras.layers.LeakyReLU(alpha =0.4)
        
        self.conv = tf.keras.layers.Conv2D(1,
                                           kernel_size=[1,1],
                                          activation='tanh')
        self.add = tf.keras.layers.Add()
        
        
    def call(self,inputs):
        x1 = self.conv1(inputs)
        x1 = self.Norm1(x1)
        x1 = self.leak1(x1)
        x2 = self.conv2(x1)
        x2 = self.Norm2(x2)
        x2 = self.leak2(x2)
        x3 = self.conv3(x2)
        x3 = self.Norm3(x3)
        x3 = self.leak3(x3)
        x4 = self.conv4(x3)
        x4 = self.Norm4(x4)
        x4 = self.leak4(x4)
        x5 = self.conv5(x4)
        x5 = self.Norm5(x5)
        x5 = self.leak5(x5)
        x6 = self.conv6(x5)
        x6 = self.Norm6(x6)
        x6 = self.leak6(x6)
        x7 = self.conv7(x6)
        x7 = self.Norm7(x7)
        x7 = self.leak7(x7)
        x8 = self.conv8(x7)
        x8 = self.Norm8(x8)
        x8 = self.leak8(x8)
        y1 = self.deconv1(x8)
        y1 = self.deNorm1(y1)
        y1 = self.deleak1(y1)
        y1 = self.add1([x7,y1])
        y2 = self.deconv2(y1)
        y2 = self.deNorm2(y2)
        y2 = self.deleak2(y2)
        y2 = self.add2([x6,y2])
        y3 = self.deconv3(y2)
        y3 = self.deNorm3(y3)
        y3 = self.deleak3(y3)
        y3 = self.add3([x5,y3])
        y4 = self.deconv4(y3)
        y4 = self.deNorm4(y4)
        y4 = self.deleak4(y4)
        y4 = self.add4([x4,y4])
        y5 = self.deconv5(y4)
        y5 = self.deNorm5(y5)
        y5 = self.deleak5(y5)
        y5 = self.add5([x3,y5])
        y6 = self.deconv6(y5)
        y6 = self.deNorm6(y6)
        y6 = self.deleak6(y6)
        y6 = self.add6([x2,y6])
        y7 = self.deconv7(y6)
        y7 = self.deNorm7(y7)
        y7 = self.deleak7(y7)
        y7 = self.add7([x1,y7])
        y8 = self.deconv8(y7)
        y8 = self.deNorm8(y8)
        y8 = self.deleak8(y8)
        x = self.conv(y8)
        output = self.add([x,inputs])
        return output


'''
class dis(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.con1 = tf.keras.layers.Conv2D(filters=32,
                                           kernel_size=[4,4],
                                           strides=(2, 2),
                                           use_bias=False,
                                           input_shape=(256,256,1))
        self.Norm1 = tf.keras.layers.BatchNormalization()
        self.leak1 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.con2 = tf.keras.layers.Conv2D(filters=32,
                                           kernel_size=[4,4],
                                          strides=(2, 2),
                                          use_bias=False)
        self.Norm2 = tf.keras.layers.BatchNormalization()
        self.leak2 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.con3 = tf.keras.layers.Conv2D(filters=32,
                                           kernel_size=[4,4],
                                          strides=(2, 2),
                                          use_bias=False)
        self.Norm3 = tf.keras.layers.BatchNormalization()
        self.leak3 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.con4 = tf.keras.layers.Conv2D(filters=32,
                                           kernel_size=[4,4],
                                          strides=(2, 2),
                                          use_bias=False)
        self.Norm4 = tf.keras.layers.BatchNormalization()
        self.leak4 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.con5 = tf.keras.layers.Conv2D(filters=32,
                                           kernel_size=[4,4],
                                          strides=(2, 2),
                                          use_bias=False)
        self.Norm5 = tf.keras.layers.BatchNormalization()
        self.leak5 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.con6 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=[4,4],
                                          strides=(2, 2),
                                          use_bias=False)
        self.Norm6 = tf.keras.layers.BatchNormalization()
        self.leak6 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.con7 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=[1,1],
                                          use_bias=False)
        self.Norm7 = tf.keras.layers.BatchNormalization()
        self.leak7 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.con8 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=[1,1],
                                          use_bias=False)
        self.Norm8 = tf.keras.layers.BatchNormalization()
        self.leak8 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.con9 = tf.keras.layers.Conv2D(filters=32,
                                           kernel_size=[1,1],
                                          use_bias=False)
        self.Norm9 = tf.keras.layers.BatchNormalization()
        self.leak9 = tf.keras.layers.LeakyReLU(alpha =0.4)
        #self.con10 = tf.keras.layers.Conv2D(filters=32,
                                           #kernel_size=[3,3])
        #self.Norm10 = tf.keras.layers.BatchNormalization()
        #self.leak10 = tf.keras.layers.LeakyReLU(alpha =0.4)
        #self.con11 = tf.keras.layers.Conv2D(filters=64,
                                           #kernel_size=[3,3])
        #self.Norm11 = tf.keras.layers.BatchNormalization()
        #self.leak11 = tf.keras.layers.LeakyReLU(alpha =0.4)
        self.flatten = tf.keras.layers.Flatten()
                
        self.dense1 = tf.keras.layers.Dense(1,
                                         activation=tf.nn.softmax)
    def call(self,inputs):
        x = self.con1(inputs)
        x = self.Norm1(x)
        x = self.leak1(x)
        x = self.con2(x)
        x = self.Norm2(x)
        x = self.leak2(x)
        x = self.con3(x)
        x = self.Norm3(x)
        x = self.leak3(x)
        x = self.con4(x)
        x = self.Norm4(x)
        x = self.leak4(x)
        x = self.con5(x)
        x = self.Norm5(x)
        x = self.leak5(x)
        x = self.con6(x)
        x = self.Norm6(x)
        x = self.leak6(x)
        x = self.con7(x)
        x = self.Norm7(x)
        x = self.leak7(x)
        x = self.con8(x)
        x = self.Norm8(x)
        x = self.leak8(x)
        x = self.con9(x)
        x = self.Norm9(x)
        x = self.leak9(x)
        #x = self.con10(x)
        #x = self.Norm10(x)
        #x = self.leak10(x)
        #x = self.con11(x)
        #x = self.Norm11(x)
        #x = self.leak11(x)
        x = self.flatten(x)
        output = self.dense1(x)
        return output
'''
    
def dis():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(256,256,1)))
    model.add(layers.Conv2D(filters=64,
                            kernel_size=[4,4],
                            strides=(2, 2),
                            padding='same',
                            use_bias=False))#1
    model.add(layers.BatchNormalization())#output = [128,128,64]
    #model.add(layers.LeakyReLU(alpha =0.4))
    model.add(layers.Conv2D(filters=128,
                            kernel_size=[4,4],
                            strides=(2, 2),
                            padding='same',
                            use_bias=False))#2
    model.add(layers.BatchNormalization())#output = [64,64,128]
    #model.add(layers.LeakyReLU(alpha =0.4))
    model.add(layers.Conv2D(filters=256,
                            kernel_size=[4,4],
                            strides=(2, 2),
                            padding='same',
                            use_bias=False))#3
    model.add(layers.BatchNormalization())#output = [32,32,256]
    #model.add(layers.LeakyReLU(alpha =0.4))
    model.add(layers.Conv2D(filters=512,
                            kernel_size=[4,4],
                            strides=(2, 2),
                            padding='same',
                            use_bias=False))#4
    model.add(layers.BatchNormalization())#output = [16,16,512]
    #model.add(layers.LeakyReLU(alpha =0.4))
    model.add(layers.Conv2D(filters=1024,
                            kernel_size=[4,4],
                            strides=(2, 2),
                            padding='same',
                            use_bias=False))#5
    model.add(layers.BatchNormalization())#output = [8,8,1024]
    #model.add(layers.LeakyReLU(alpha =0.4))
    model.add(layers.Conv2D(filters=1024,
                            kernel_size=[4,4],
                            strides=(2, 2),
                            padding='same',
                            use_bias=False))#6#output = [4,4,1024]
    model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha =0.4))
    model.add(layers.Conv2D(filters=1024,
                            kernel_size=[1,1],
                            strides=(2, 2),
                            padding='same',
                            use_bias=False))#7
    model.add(layers.BatchNormalization())#output = [2,2,1024]
    #model.add(layers.LeakyReLU(alpha =0.4))
    model.add(layers.Conv2D(filters=512,
                            kernel_size=[1,1],
                            strides=(2, 2),
                            padding='same',
                            use_bias=False))#8
    model.add(layers.BatchNormalization())#output = [1,1,512]
    #model.add(layers.LeakyReLU(alpha =0.4))
    model.add(layers.Conv2D(filters=128,
                            kernel_size=[1,1],
                            strides=(1, 1),
                            padding='same',
                            use_bias=False))#9
    model.add(layers.BatchNormalization())#output = [1,1,128]
    model.add(layers.Conv2D(filters=128,
                            kernel_size=[1,1],
                            strides=(1, 1),
                            padding='same',
                            use_bias=False))#10
    model.add(layers.BatchNormalization())#output = [1,1,128]
    model.add(layers.Flatten())
    model.add(layers.Dense(1,
                            activation=tf.nn.sigmoid))
    
    return model

if __name__ == "__main__":
    pass