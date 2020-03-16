import sys
import os
import imageio
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import cv2

# 用于读取图像
def load_input_image(path, is_show=False):
    print("in load_input_image() function")
    # 获取绝对路径
    fname = sys.path[0] + path
    print("\t Image path: [" + fname + "]")

    # 判断是否为图像
    if not os.path.isfile(fname):
        print("\t Open image failed, please check path")
        return -1

    # 读取图像
    image = imageio.imread(fname)
    print("\t Load success!")
    print("\t Image shape:" + str(image.shape))

    # 展示图像
    if is_show:
        plt.imshow(image)
        plt.show()
    return image

def blur_noise_process(image, kernel, scale_factor, output_shape, noise_std):
    out_im = np.zeros_like(image)

    for channel in range(0, out_im.shape[2]):
        out_im[:, :, channel] = filters.correlate(image[:, :, channel], kernel)

    # 使用等差数列的方式缩小分辨率
    res = out_im[np.round(np.linspace(0, image.shape[0] - 1 / scale_factor, output_shape[0])).astype(int)[:, None],
                 np.round(np.linspace(0, image.shape[1] - 1 / scale_factor, output_shape[1])).astype(int), :]

    # 添加高斯噪声（0, 0.0125）
    res = np.clip(res + np.random.randn(*res.shape) * noise_std, 0, 1)

    return res



# 对图像进行放大缩小，默认为 0.5 倍
def imresize(image, is_show = False, scale_factor = None, kernel = None, output_shape = None, noise_std = 0.0):
    # 计算随机后的 LR 分辨率
    if scale_factor is not None:
        image_shape = np.array(np.array(image.shape)[:-1] * scale_factor, dtype=int)
    elif output_shape is not None:
        scale_factor = output_shape[0] / image.shape[0]
        image_shape = output_shape[:-1]
    else:
        print("scale_factor 和 output_shape 均无效")

    if type(kernel) == np.ndarray:
        return blur_noise_process(image, kernel, scale_factor, image_shape, noise_std)


    # resize 处理
    method = {
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
        "linear": cv2.INTER_LINEAR,
        None: cv2.INTER_CUBIC
    }.get(kernel)
    LR_image = cv2.resize(image, tuple(image_shape[::-1]), method) # cv2 的 w 和 h 与 image.shape 相反，因此使用 image_shape[::-1]) 反向


    # 展示放缩后的图像
    if is_show:
        plt.imshow(LR_image)
        plt.show()

    # print()
    return LR_image



# 对图像进行随机裁剪
def random_crop(image, is_show = False, size = None):
    # print("in random_crop function")
    # 若图像尺寸小于输入的 crop 尺寸 或者 启用了默认 crop 尺寸，则使用默认定义的尺寸
    default_size = 0.23
    # 获取原图数据的 ndarray 类型，方便计算
    ori_shape = np.array(image.shape)
    # print("\t ori_shape = %s" % (ori_shape))

    # 确定裁剪图像尺寸
    if size is None:
        # print('size = None')
        size = ori_shape[:-1] * default_size
    else:
        # print('size is not None')
        size = size * np.array(image.shape[:-1])

    size = size.astype(np.int16)
    # print('\t size.shape = ' + str(size))

    # 随机 crop 图像左上角的位置
    random_range = ori_shape[:-1] - size
    # print('\t random_range = %s' %(random_range))
    s_w = random.sample(range(0, random_range[1]), 1)
    s_h = random.sample(range(0, random_range[0]), 1)
    # print('\t w = %s, h = %s' %(s_w, s_h))

    crop_image = image[s_h[0]:s_h[0]+size[0], s_w[0]:s_w[0]+size[1], :]
    # print('\t '+ str(crop_image.shape))

    if is_show:
        plt.imshow(crop_image)
        plt.show()

    # print()
    return crop_image;

# 进行反投影
def back_projection(sr_img, lr_img, down_kernel, up_kernel):

    sr_img += imresize(
        lr_img - imresize(sr_img, False, None, down_kernel, lr_img.shape),
        False, None, up_kernel, sr_img.shape
    )
    return np.clip(sr_img, 0, 1)