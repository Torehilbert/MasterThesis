import numpy as np
import cv2



def append_degraded_channels(image, degrade_discretize=None, degrade_blur=None, degrade_noise=None):
    # Find number of output channels
    n_channels_out = 1
    for l in [degrade_discretize, degrade_blur, degrade_noise]:
        if l is not None:
            n_channels_out += len(l)

    image_out = np.empty(shape=(image.shape[0], image.shape[1], n_channels_out), dtype="float32")
    
    image_out[:,:,0] = image
    cursor = 1
    for disc in degrade_discretize:
        image_out[:,:,cursor] = degrade_to_levels_new(image, disc, dtype=np.float32)
        cursor += 1
    
    for bsize in degrade_blur:
        image_out[:,:,cursor] = degrade_by_blurring(image, bsize)
        cursor += 1

    for nlevel in degrade_noise:
        image_out[:,:,cursor] = degrade_by_noise(image, nlevel)
        cursor += 1
    
    return image_out


def degrade_by_noise(image, noise_std):
    return (image + noise_std * np.random.randn(image.shape[0], image.shape[1]))/(noise_std/2)


def degrade_by_blurring(image, blur_size, mask_by_zero=False):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(blur_size, blur_size))/(blur_size*blur_size)
    image_blurred = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    if mask_by_zero:
        mask = image==0
        background = mask==1
        image_blurred[background] = image[background]
    return image_blurred


def degrade_to_levels(image, levels, dtype):
    image = np.array(image, dtype=np.float64)
    levels = np.array(levels)

    dists = []
    for level in levels:
        dists.append(np.abs(level * np.ones_like(image) - image))

    dists = np.stack(dists, axis=2)
    return np.array(levels[np.argmin(dists, axis=2)], dtype=dtype)


def degrade_to_levels_new(image, num_levels, dtype):
    mask_idx_cell = np.where(image != 0)
    pixels = image[mask_idx_cell[0], mask_idx_cell[1]]
    mean = np.mean(pixels)
    std = np.std(pixels)
    pixels = np.round(num_levels * (pixels - mean)/std)
    pixels = pixels*std + mean

    image = np.array(image, dtype=dtype)
    image[mask_idx_cell[0], mask_idx_cell[1]] = pixels
    return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = r"D:\Speciale\Data\Dummy_dataset\Xcyto10-4x\crops\images\images_BrightField\20200702-00001_1_1.npy"

    image = np.load(image_path)
    image_min = np.min(image)
    image_max = np.max(image)

    n_rows = 3
    n_cols = 5

    num_levels = [1, 0.5, 0.35]
    blur_sizes = [6, 8, 16]
    noise_levels = [2,4,8]

    plt.figure()
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(image, cmap='gray')

    for i in range(0, len(num_levels)):
        n_lev = num_levels[i]
        plt.subplot(n_rows,n_cols, i+2)
        plt.imshow(degrade_to_levels_new(image, n_lev, dtype=np.float32), cmap='gray', vmin=image_min, vmax=image_max)  

    for i in range(len(blur_sizes)):
        bsize = blur_sizes[i]
        plt.subplot(n_rows,n_cols, i+2 + n_cols)
        plt.imshow(degrade_by_blurring(image, bsize), cmap='gray', vmin=image_min, vmax=image_max)    

    for i in range(len(noise_levels)):
        nlevel = noise_levels[i]
        plt.subplot(n_rows, n_cols, i+2 + 2*n_cols)
        plt.imshow(degrade_by_noise(image, noise_levels[i]), cmap='gray')
    plt.show()