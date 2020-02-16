import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import math
from skimage import measure

def Round(num):
    res = round(num)
    if(res > num and num - int(num) == 0.5):
        res -= 1
    return res

def NN_interpolation(image, scale):
    image = np.array(image)
    new_width = Round(image.shape[0] * scale);
    new_height = Round(image.shape[1] * scale);
    new_shape = image.shape[2];

    scale = 1 / scale;
    newImage = np.zeros((new_width, new_height, new_shape), dtype=np.uint8)
    for i in range(newImage.shape[0]):
        for j in range(newImage.shape[1]):
            new_i = Round(i * scale)
            new_j = Round(j * scale)
            newImage[i, j] = image[new_i, new_j]
    return newImage

def bilinear_interpolation(image, scale):
    image = np.array(image)

    new_width = Round(image.shape[0] * scale);
    new_height = Round(image.shape[1] * scale);
    new_shape = image.shape[2];

    scale = 1.0 / scale;
    newImage = np.zeros((new_width, new_height, new_shape), dtype=np.uint8)

    for i in range(newImage.shape[0]):
        for j in range(newImage.shape[1]):
            new_i = i * scale
            new_j = j * scale

            integral_i = int(new_i)
            fraction_i = new_i - integral_i
            integral_j = int(new_j)
            fraction_j = new_j - integral_j

            if(integral_i + 1 == image.shape[0] or integral_j + 1 == image.shape[1]):
                newImage[i, j] = image[integral_i, integral_j]
                continue;

            # interpolation_AB = image[integral_i, integral_j]        * (1.0 - fraction_i) + image[integral_i + 1, integral_j]        * fraction_i
            # interpolation_CD = image[integral_i, integral_j + 1]    * (1.0 - fraction_i) + image[integral_i + 1, integral_j + 1]    * fraction_i
            # Final = interpolation_AB * (1.0 - fraction_j) + interpolation_CD * fraction_j

            Final =     (1.0 - fraction_i) * (1.0 - fraction_j) * image[integral_i, integral_j] + \
                        (1.0 - fraction_i) * fraction_j         * image[integral_i, integral_j + 1] + \
                        fraction_i         * (1.0 - fraction_j) * image[integral_i + 1, integral_j] + \
                        fraction_i         * fraction_j         * image[integral_i + 1, integral_j + 1]

            newImage[i, j] = Final

    return newImage

def PSNR(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)

    MSE = np.mean((image1 / 255.0 - image2 / 255.0) ** 2)
    if(MSE < 1.0e-10):
        return 100.0

    return 20 * math.log10(1.0 / math.sqrt(MSE))
    return psnr



image = Image.open('img/img3.png')
# image.save('compare/origin.png')
# plt.imshow(image);
# plt.show()

NN_interpolation_image = Image.fromarray(NN_interpolation(image, 2))
bilinear_interpolation_image = Image.fromarray(bilinear_interpolation(image, 2))
print(PSNR(np.array(NN_interpolation_image), np.array(bilinear_interpolation_image)))
print(measure.compare_psnr(np.array(NN_interpolation_image), np.array(bilinear_interpolation_image)))



# plt.imshow(NN_interpolation_image)
# plt.show()
#
# plt.imshow(bilinear_interpolation_image)
# plt.show()
