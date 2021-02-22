import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_channels(image, save_path=None, show=True, normalize_across_channels=True, vmin=None, vmax=None, cmap='gray', channel_names=None, figsize=(6,4)):
    if normalize_across_channels:
        vmin = np.min(image)
        vmax = np.max(image)
    
    channel_axis = 3 if len(image.shape) == 4 else 2

    subplot_dims = _get_subplot_dims(image.shape[channel_axis])

    plt.figure(figsize=figsize)
    for i in range(image.shape[channel_axis]):
        plt.subplot(subplot_dims[0],subplot_dims[1],i+1)
        plt.imshow(image[:,:,i], cmap=cmap, vmin=vmin, vmax=vmax)
        if channel_names is not None and len(channel_names) > i:
            plt.title(channel_names[i])
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])

    plt.tight_layout()

    if save_path is not None:
        if os.path.isdir(os.path.dirname(save_path)):
            plt.savefig(save_path)
        else:
            print("ERROR: Something is wrong with save_path: %s - ignoring..." % save_path)

    if show:
        plt.show()
    
    plt.clf()
    plt.close()



def _get_subplot_dims(n):
    if n==1:
        return (1,1)
    elif n==2:
        return (1,2)
    elif n==3:
        return (1,3)
    elif n==4:
        return (1,4)
    elif n==5 or n==6:
        return (2,3)
    elif n==7 or n==8:
        return (2,4)
    elif n==9 or n==10:
        return (2,5)
    elif n>=11 and n<=15:
        return (3,5)
    elif n>=16 and n<=20:
        return (4,5)
    elif n>=21 and n<=25:
        return (5,5)
    elif n>=26:
        raise Exception("Don't you think a subplot with more than 26 entries is way too messy?")
    


if __name__ == "__main__":
    image = np.zeros(64,64,10)
    
    for i in range(image.shape[2]):
        image[:,:,i] = i

    visualize_channels(image)
    