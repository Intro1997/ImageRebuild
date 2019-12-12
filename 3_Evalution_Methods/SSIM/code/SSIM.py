from PIL import Image
from skimage import measure
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.util.dtype import dtype_range

def SSIM(image1, image2):
    L = 255
    K1 = 0.01
    K2 = 0.03

    # 因为是均值滤波，如果是高斯滤波，则依照论文中应设置为 11
    win_size = 7
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # E(image1)  E(image2) 非高斯滤波滑窗尺寸为 7
    mu1 = uniform_filter(image1, size=win_size)
    mu2 = uniform_filter(image2, size=win_size)

    # E(image1^2) E(image2^2)
    mu11 = uniform_filter(image1 * image1, size=win_size)
    mu22 = uniform_filter(image2 * image2, size=win_size)

    # E(iamge1 * image2)
    mu12 = uniform_filter(image1 * image2, size=win_size)

    # 像素点总量 = 滑窗尺寸 * 图像维度
    NP = win_size ** image1.ndim

    # 采用无偏估计，分母 = 1 / （N-1）
    denominator = NP / (NP - 1.0)
    # sigma_1^2
    sigma1 = denominator * (mu11 - mu1 * mu1)
    # sigma_2^2
    sigma2 = denominator * (mu22 - mu2 ** 2)
    # sigma_12
    sigma12 = denominator * (mu12 - mu1 * mu2)

    # light compare function
    C1 = (K1 * L) ** 2;
    C2 = (K2 * L) ** 2;

    SSIM = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    # 截去边缘部分，因为卷积得到的边缘部分的均值并不准确，是靠扩充边缘像素的方式得到的。
    pad = (win_size - 1) // 2

    SSIM = SSIM[pad:SSIM.shape[0] - pad]
    S = SSIM[:, pad:SSIM.shape[1] - pad]
    mssim = SSIM.mean()

    return mssim

o = Image.open("img/origin.png")
b = Image.open("img/blur.png")
c = Image.open('img/compressed.png')

print(SSIM(np.array(o), np.array(c)))
print(measure.compare_ssim(np.array(o), np.array(c), multichannel=True))
