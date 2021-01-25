from scipy.fftpack import fft2,fftshift,ifftshift,ifft2
from scipy.io import loadmat, savemat
import numpy as n
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import math
import numpy as n
import imageio
from define_genewithbias import dis,gene,to_bad_img,fft_abs_for_map_fn
import os
import sys
import skimage.metrics as sk
#import matplotlib.pyplot as p

def main_train():
    global file_name
    global mask_name
    global rate
    global loss
    f_name = file_name

    train_value_txt = 'train_value_'+f_name+'.txt'
    test_value_txt = 'test_value_'+f_name+'.txt'
    os.mkdir(f_name)
    os.mkdir(f_name + '/val_img')
    os.mkdir(f_name + '/train_img')
    os.mkdir(f_name + '/test_img')

    if (mask_name == 'Gaussian2D'):
        path = 'mask/'+ mask_name +'/'
        file_name = os.listdir(path)
        mask = []
        for i in file_name:
            maskfl = loadmat(path+i)
            print(maskfl)
            mask.append(maskfl['maskRS2'])
    elif (mask_name == 'Gaussian1D'):
        path = 'mask/'+ mask_name +'/'
        file_name = os.listdir(path)
        mask = []
        for i in file_name:
            maskfl = loadmat(path+i)
            print(maskfl)
            mask.append(maskfl['maskRS1'])
    else:
        path = 'mask/'+ mask_name +'/'
        file_name = os.listdir(path)
        mask = []
        for i in file_name:
            maskfl = loadmat(path+i)
            print(maskfl)
            mask.append(maskfl['population_matrix'])

    #######################################################################################################
    batch_size = 20

    gene1 = gene()
    dis1 = dis()

    vgg16 = tf.keras.applications.VGG16(weights='imagenet')
    num = 0
    learning_rate1 = 0.0001
    decay_rate = 0.8
    dis_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate1, beta_1=0.1)
    gene_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate1, beta_1=0.1)

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gene = train_datagen.flow_from_directory(
        'data/',
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gene = test_datagen.flow_from_directory(
        'data1/',
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None)



    shape = tf.TensorSpec(shape = (batch_size,256,256,1))# 设置模型的输入
    gene1._set_inputs(shape)




    ######################################################################################################## train start


    for x in train_gene:

        per_x = n.concatenate((n.concatenate((x,x),3),x),3)
        per_x = per_x[:,15:239,15:239,:]
        per_x = tf.convert_to_tensor(per_x)
        per_loss_x = vgg16(per_x)
        x_u = []
        if(rate=='10'):
            for i in range(x.shape[0]):
                pic = to_bad_img(x[i,:,:,:],mask[1])
                x_u.append(pic)
        elif(rate=='20'):
            for i in range(x.shape[0]):
                pic = to_bad_img(x[i,:,:,:],mask[2])
                x_u.append(pic)
        else:
            for i in range(x.shape[0]):
                pic = to_bad_img(x[i,:,:,:],mask[6])
                x_u.append(pic)
        x_u = n.array(x_u)#降采样图像
        x = n.array(x)#原本的mri图像
        x_u = tf.convert_to_tensor(x_u)
        x = tf.convert_to_tensor(x)
        #true_label = n.ones([x.shape[0],1])
        #false_label = n.zeros([x.shape[0],1])
        #to_dis_label = n.concatenate((false_label, true_label),0)#用来训练辨别器的真实标签


        with tf.GradientTape() as tape_gene, tf.GradientTape() as tape_dis:
            x_r = gene1(x_u, training=True)#重建图片x_r
            #for i in range(x_r.shape[0]):
                #a = x_r[i,:,:,:]-tf.math.reduce_min(x_r[i,:,:,:]).numpy()
                #b = tf.math.reduce_max(x_r[i,:,:,:]).numpy()-tf.math.reduce_min(x_r[i,:,:,:]).numpy()
                #x_r[i,:,:,:] = a/b
            #print(tf.math.reduce_max(x_r))
            #print(tf.math.reduce_min(x_r))

            pred_label_x_r = dis1(x_r, training=True)#重建图片的预测标签
            pred_label_x = dis1(x, training=True)#真实图片的预测标签
            true_label_x_r = 0.95*tf.ones_like(pred_label_x)#用来训练辨别器的真实标签
            true_label_x = 0.05*tf.ones_like(pred_label_x_r)#tf.zeros_like(pred_label_x_r)#用来训练辨别器的真实标签
            nmse_a = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(x_r, x), axis=[1, 2, 3]))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1, 2, 3]))
            g_nmse = tf.reduce_mean(nmse_a / nmse_b)#nmse loss

            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(x, x_r, max_val=1))

            l1 = tf.keras.losses.MeanAbsoluteError()
            l1_loss = l1(x_r,x)#L1 loss
            print(l1_loss.numpy(),pred_label_x_r, pred_label_x,'###############################')
            per_x_r = n.concatenate((n.concatenate((x_r,x_r),3),x_r),3)
            per_x_r = per_x_r[:,15:239,15:239,:]
            per_x_r = tf.convert_to_tensor(per_x_r)
            per_loss_x_r = vgg16(per_x_r) #perceptual loss
            print(per_loss_x_r.shape,per_loss_x.shape, tf.reduce_mean(per_loss_x_r).numpy(), tf.reduce_mean(per_loss_x).numpy(),
                  '##########################################')
            per_loss = tf.reduce_mean(tf.math.squared_difference(per_loss_x_r,per_loss_x))

            ft = tf.map_fn(fft_abs_for_map_fn, x)
            fr = tf.map_fn(fft_abs_for_map_fn, x_r)
            f_loss = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.squared_difference(ft, fr), axis=[1, 2]))

            binary_loss = tf.keras.losses.BinaryCrossentropy()
            D_loss = binary_loss(true_label_x*0,pred_label_x_r) #训练生成器的辨别器loss
            if (loss == 'o'):
                to_gene_loss = D_loss + 20*g_nmse + 0.025*per_loss + 0.1*f_loss #训练生成器的总loss
            elif (loss == 'ol'):
                to_gene_loss = D_loss + 30*l1_loss + 0.025*per_loss + 0.1*f_loss #训练生成器的总loss
            elif (loss == 'ons'):
                to_gene_loss = D_loss + 20*g_nmse + 2*ssim_loss + 0.025*per_loss + 0.1*f_loss #训练生成器的总loss
            elif (loss == 'ols'):
                to_gene_loss = D_loss + 30*l1_loss + 2*ssim_loss + 0.025*per_loss + 0.1*f_loss #训练生成器的总loss
            else:
                to_gene_loss = D_loss + 20*g_nmse + 30*l1_loss + 2*ssim_loss + 0.025*per_loss + 0.1*f_loss #训练生成器的总loss

            binary_loss = tf.keras.losses.BinaryCrossentropy()
            b_loss = binary_loss(pred_label_x_r,true_label_x_r)+binary_loss(pred_label_x,true_label_x)#训练辨别器的二元交叉损失
            print('binary_loss: %f, G_loss: %f'%(b_loss.numpy(),to_gene_loss.numpy()))

            print("num: %d , nmse: %f:, L1_loss: %f, D_loss: %f , SSIM_loss = %f , perceptual_loss: %f, G_loss: %f, f_loss: %f" % (num, g_nmse.numpy(),l1_loss.numpy(),
                                                                                                     D_loss.numpy(),ssim_loss.numpy(),per_loss.numpy(),
                                                                                                     to_gene_loss.numpy(),f_loss.numpy()))


            ssim = 0
            psnr = 0
            if num%1000 ==0:
                #gene1.save_weights('save_model/gene1_checkpoint')
                #dis1.save_weights('save_model/dis_checkpoint')
                epoch = str(num//1000)
                dis_optimizer.lr  = learning_rate1 * (decay_rate**((num//1000)//2))  #减轻后期识别器学习率
                print(dis_optimizer.lr.numpy())
                filename = f_name+'/train_img/' + 'epoch' +epoch
                os.mkdir(filename)
                for i in range(x_r.shape[0]):
                    name = str(i)
                    fname_gene = 'epoch'+epoch+'_'+name+'gene_img.jpg'
                    imageio.imwrite(filename + '/' + fname_gene, x_r[i,:,:,:])
                    fname_under = 'epoch'+epoch+'_'+name+'under_img.jpg'
                    imageio.imwrite(filename + '/' + fname_under, x_u[i,:,:,:])
                    fname_good = 'epoch'+epoch+'_'+name+'good_img.jpg'
                    imageio.imwrite(filename + '/' + fname_good, x[i,:,:,:])
                    ssim = ssim + sk.structural_similarity(x.numpy()[i,:,:,0],x_r.numpy()[i,:,:,0])
                    psnr = psnr + sk.peak_signal_noise_ratio(x.numpy()[i,:,:,0],x_r.numpy()[i,:,:,0])
                    print('train_ssim: %f, train_psnr: %f'%(ssim,psnr))

                ssim = ssim/x_r.shape[0]
                psnr = psnr/x_r.shape[0]
                file = open(train_value_txt,'a')
                word = ['epoch: ', epoch ,', D_loss: ', str(b_loss.numpy()),', G_loss: ', str(to_gene_loss.numpy()),
                        ', train_NMSE: ',str(g_nmse.numpy()),' , ssim_loss: ', str(ssim_loss.numpy()) ,', train_SSIM: ', str(ssim), ', train_PSNR: ',str(psnr),'\n']
                file.writelines(word)
                file.close()
                ###validation
                for x in test_gene:
                    x_u = []
                    if(rate=='10'):
                        for i in range(x.shape[0]):
                            pic = to_bad_img(x[i,:,:,:],mask[1])
                            x_u.append(pic)
                    elif(rate=='20'):
                        for i in range(x.shape[0]):
                            pic = to_bad_img(x[i,:,:,:],mask[2])
                            x_u.append(pic)
                    else:
                        for i in range(x.shape[0]):
                            pic = to_bad_img(x[i,:,:,:],mask[6])
                            x_u.append(pic)                
                    x_u = n.array(x_u)#降采样图像
                    x = n.array(x)#原本的mri图像
                    x_u = tf.convert_to_tensor(x_u)
                    x = tf.convert_to_tensor(x)     

                    x_r = gene1(x_u, training=True)
                    nmse_a = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(x_r, x), axis=[1, 2, 3]))
                    nmse_b = tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1, 2, 3]))
                    g_nmse = tf.reduce_mean(nmse_a / nmse_b)#nmse loss
                    ssim = 0
                    psnr = 0
                    filename = f_name+'/val_img/' + 'epoch' +epoch
                    os.mkdir(filename)
                    for i in range(x_r.shape[0]):
                        name = str(i)
                        fname_gene = 'epoch'+epoch+'_'+name+'gene_img.jpg'
                        imageio.imwrite(filename + '/' + fname_gene, x_r[i,:,:,:])
                        fname_under = 'epoch'+epoch+'_'+name+'under_img.jpg'
                        imageio.imwrite(filename + '/' + fname_under, x_u[i,:,:,:])
                        fname_good = 'epoch'+epoch+'_'+name+'good_img.jpg'
                        imageio.imwrite(filename + '/' + fname_good, x[i,:,:,:])
                        ssim = ssim + sk.structural_similarity(x.numpy()[i,:,:,0],x_r.numpy()[i,:,:,0])
                        psnr = psnr + sk.peak_signal_noise_ratio(x.numpy()[i,:,:,0],x_r.numpy()[i,:,:,0])
                    ssim = ssim/x_r.shape[0]
                    psnr = psnr/x_r.shape[0]
                    print('val_nmse: %f, val_ssim: %f, val_psnr: %f'%(g_nmse.numpy(),ssim,psnr))
                    file = open(train_value_txt,'a')
                    word = ['epoch: ', epoch ,', val_NMSE: ',str(g_nmse.numpy()), ', val_SSIM: ', str(ssim), ', val_PSNR: ',str(psnr),'\n']
                    file.writelines(word)
                    file.close()
                    break


        grads_gene = tape_gene.gradient(to_gene_loss , gene1.trainable_variables)
        grads_dis = tape_dis.gradient(b_loss , dis1.trainable_variables)
        gene_optimizer.apply_gradients(grads_and_vars=zip(grads_gene , gene1.trainable_variables))
        dis_optimizer.apply_gradients(grads_and_vars = zip(grads_dis , dis1.trainable_variables))

        num = num+1
        if num > 30000:
            break






    gene1.save_weights('save_model/generator_model_'+name)
    dis1.save('save_model/discriminator_model_' + name +'.h5')


    ############################################################################# test start

    epoch = 0
    for x in test_gene:
        x_u = []
        if(rate=='10'):
            for i in range(x.shape[0]):
                pic = to_bad_img(x[i,:,:,:],mask[1])
                x_u.append(pic)
        elif(rate=='20'):
            for i in range(x.shape[0]):
                pic = to_bad_img(x[i,:,:,:],mask[2])
                x_u.append(pic)
        else:
            for i in range(x.shape[0]):
                pic = to_bad_img(x[i,:,:,:],mask[6]) 
                x_u.append(pic)
                
        
        x_u = n.array(x_u)#降采样图像
        x = n.array(x)#原本的mri图像
        x_u = tf.convert_to_tensor(x_u)
        x = tf.convert_to_tensor(x)    

        x_r = gene1(x_u)
        nmse_a = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(x_r, x), axis=[1, 2, 3]))
        nmse_b = tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1, 2, 3]))
        g_nmse = tf.reduce_mean(nmse_a / nmse_b)#nmse loss
        ssim = 0
        psnr = 0
        for i in range(x_r.shape[0]):
            name = str(i)
            ep = str(epoch)
            fname_gene = 'epoch'+ep+'_'+name+'gene_img.jpg'
            imageio.imwrite(f_name + '/test_img/'+fname_gene, x_r[i,:,:,:])
            fname_under = 'epoch'+ep+'_'+name+'under_img.jpg'
            imageio.imwrite(f_name + '/test_img/'+fname_under, x_u[i,:,:,:])
            fname_good = 'epoch'+ep+'_'+name+'good_img.jpg'
            imageio.imwrite(f_name + '/test_img/'+fname_good, x[i,:,:,:])
            ssim = ssim + sk.structural_similarity(x.numpy()[i,:,:,0],x_r.numpy()[i,:,:,0])
            psnr = psnr + sk.peak_signal_noise_ratio(x.numpy()[i,:,:,0],x_r.numpy()[i,:,:,0])
        ssim = ssim/x_r.shape[0]
        psnr = psnr/x_r.shape[0]
        file = open(test_value_txt,'a')
        word = ['epoch: ', ep ,', test_NMSE: ',str(g_nmse.numpy()), ', test_SSIM: ', str(ssim), ', test_PSNR: ',str(psnr),'\n']
        file.writelines(word)
        file.close()

        epoch = epoch + 1
        if epoch > 10:
            break



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file_name', type=str, default='gan', help='gan_10%')
    parser.add_argument('--mask_name', type=str, default='Gaussian2D', help='Gaussian1D, Gaussian2D, Poisson2D')
    parser.add_argument('--rate', type=str, default='30', help='10,20,50')
    parser.add_argument('--loss', type=str, default='olns', help='o,ol,ons,ols,olns')
    
    args = parser.parse_args()
    global file_name
    file_name = args.file_name
    global mask_name
    mask_name = args.mask_name
    global rate
    rate = args.rate
    global loss
    loss = args.loss
    
    main_train()