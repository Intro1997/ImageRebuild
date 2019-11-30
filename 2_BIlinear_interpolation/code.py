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

            interpolation_AB = image[integral_i, integral_j]        * (1.0 - fraction_i) + image[integral_i + 1, integral_j]        * fraction_i
            interpolation_CD = image[integral_i, integral_j + 1]    * (1.0 - fraction_i) + image[integral_i + 1, integral_j + 1]    * fraction_i
            Final = interpolation_AB * (1.0 - fraction_j) + interpolation_CD * fraction_j

            # Final =     (1.0 - fraction_i) * (1.0 - fraction_j) * image[integral_i, integral_j] + \
            #             (1.0 - fraction_i) * fraction_j         * image[integral_i, integral_j + 1] + \
            #             fraction_i         * (1.0 - fraction_j) * image[integral_i + 1, integral_j] + \
            #             fraction_i         * fraction_j         * image[integral_i + 1, integral_j + 1]

            newImage[i, j] = Final

    return newImage
