


#Deprecated for latest verison

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise
import scipy.ndimage as ndi
from skimage import  color


# Load the image
img = cv2.imread("test1.png")

# Add salt-and-pepper noise to the image.
noise_img = random_noise(img, mode='gaussian', mean=0.05, var=0.05)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img = np.array(255*noise_img, dtype = 'uint8')
image = color.rgb2gray(img)
blurred = ndi.uniform_filter(image, size=5, mode='mirror')
img_stack = np.stack(blurred)



# Display the noise image
cv2.imshow('blur', blurred)
cv2.waitKey(0)



















#
# img = cv2.imread("test1.png")
#
# #    ,   0,02
# out1 = sp_noise(img, prob=0.2)
#
# # Добавить гауссовский шум, среднее значение 0, а дисперсия составляет 0,01
# out2 = gasuss_noise(img, mean=0.3, var=0.3)
#
#
# # Отображать изображение
# titles = ['Original Image', 'Add Salt and Pepper noise','Add Gaussian noise']
# images = [img, out1, out2]
#
# plt.figure(figsize = (20, 15))
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()