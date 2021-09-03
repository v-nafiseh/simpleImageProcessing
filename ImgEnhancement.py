from PIL import Image
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape, shape
import numpy as np
from skimage.color.colorconv import gray2rgb
from skimage.transform import rescale
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma,denoise_wavelet, estimate_sigma
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage import data, img_as_float ,io, exposure , color
from skimage.util import random_noise
import cv2
import os
import matplotlib

def read_image(path):
        data=[]
        imageformat=".png"
        imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
        for el in imfilelist:
                image = cv2.imread(el)
                data.append(image)
        return data

#########################################################

# rescaling:
def scaling(data_img, index):
        original_img = data_img
        rescaled_img = rescale(original_img, 0.85, anti_aliasing=False)  
        # print(rescaled_data[1].shape)
        plt.imshow(rescaled_img , cmap = 'gray')
        plt.axis('off')        
        plt.show()
        
        path = "./rescale"
        img = cv2.convertScaleAbs(rescaled_img, alpha=(250.0))
        cv2.imwrite(os.path.join(path, "rescailed-{}.png".format(index+1)  ),img)

org_data = read_image("./original")

for index,i in enumerate(org_data):
        scaling(i,index)

#########################################################

# contrast v1:

def contrast(data_img, index):
        original_img = data_img
        gamma_contrast = exposure.adjust_gamma(original_img, 3)
        plt.imshow(gamma_contrast, cmap = 'gray')
        plt.axis('off')
        plt.show()

        path= "./contrast1"
        # img = cv2.convertScaleAbs(gamma_contrast, alpha=(3.0))
        # cv2.imwrite(os.path.join(path, "contrasted1-{}.png".format(index+1)  ),img) 
        plt.imsave(os.path.join(path, "contrasted1-{}.png".format(index+1)),gamma_contrast)

rescaled_data = read_image("./rescale")

for index,i in enumerate(rescaled_data):
        contrast(i , index)

#########################################################

# contrast v2:
 
def contrast2(data_img,index):
        original_img = data_img
        p1,p2 = np.percentile( original_img , (8, 92)) 
        contrasted_data = exposure.rescale_intensity( original_img , in_range=( p1, p2 ))
        plt.imshow(contrasted_data, cmap = 'gray')
        plt.axis('off')
        plt.show()

        path= "./contrast2"
        plt.imsave(os.path.join(path, "contrasted2-{}.png".format(index+1)),contrasted_data)

path= "./contrast1"
cont1_data = read_image(path)

for index,i in enumerate(cont1_data):
        contrast2(i,index)

#########################################################

#denoising:

def denoising( index,img_data ):
        original = img_as_float(img_data)
        sigma = 0.1
        noisy = random_noise(original, var=sigma**2)

        sigma_est = np.mean(estimate_sigma(noisy, average_sigmas=True, multichannel=True))
        print(f"Estimated Gaussian noise standard deviation = ${sigma_est}")
        denoised_img = denoise_wavelet(noisy, rescale_sigma=True)
        plt.imshow(denoised_img)
        plt.axis('off')
        plt.show()

        path="./denoise"
        img = cv2.convertScaleAbs(denoised_img, alpha=(250.0))
        cv2.imwrite(os.path.join(path, "denoised-{}.png".format(index+1) ),img)

path="./contrast2"        
cont2_data = read_image(path)

for index, i in enumerate(cont2_data):
        denoising(index , i) 

##################################################

# thresholding v1

def thresholding(img_data, index):
        original_img = color.rgb2gray(img_data)
        thresh = threshold_otsu(original_img)
        binary = original_img > thresh

        plt.imshow(binary, cmap=plt.cm.gray)
        plt.title('Thresholded')
        plt.axis('off')
        plt.show()

        path="./thresholder1"
        # plt.imsave(os.path.join(path, "thresholdered1-{}.png".format(index+1)),binary)
        im = Image.fromarray(binary)
        im.save(os.path.join(path,"thresholdered1-{}.png".format(index+1)))

denoised_data = read_image(path="./denoise")
for index, i in enumerate(denoised_data):
        thresholding(i , index)

#######################################################

# # thresholding v2

def thresholding2(img_data,index):

        original_img = color.rgb2gray(img_data)
        # binary_niblack = original_img > threshold_niblack(original_img, k=0.8)
        binary_sauvola = original_img > threshold_sauvola(original_img)

        plt.imshow(binary_sauvola, cmap=plt.cm.gray)
        plt.title('Sauvola Threshold')
        plt.axis('off')
        plt.show()

        path="./thresholder2"
        im = Image.fromarray(binary_sauvola)
        im.save(os.path.join(path,"thresholdered2-{}.png".format(index+1)))

thresh1_data = read_image(path="./thresholder1")

for index, i in enumerate(thresh1_data):
        thresholding2(i , index)

##########################################################
