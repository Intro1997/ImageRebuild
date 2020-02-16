import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

image = Image.open('img/img3.png')
plt.imshow(image);
plt.show()

image = Image.fromarray(NN_interpolation(image, 5))

plt.imshow(image);
plt.show()
image.save('res/res.png')



