import numpy as np


def create_channels_images(image, f1, f2):
    out = np.zeros(shape=(image.shape[0], image.shape[1], 2), dtype=image.dtype)
    inds = np.where(image!=0)
    out[inds[0], inds[1],0] = f1
    out[inds[0], inds[1],1] = f2
    return out


def get_features(classes, number_of_classes=2, multiplier=1):
    if number_of_classes==2:
        return _get_features_k2(classes)
    elif number_of_classes==3:
        return _get_features_k3(classes, multiplier=multiplier)
    else:
        raise Exception("Does not support other than 2 or 3 classes!")


def _get_features_k2(classes):
    N = len(classes)

    y_possibles = np.array([[0,1],[1,0]])
    x_c = np.random.randint(0, 2, size=(N,))
    y_c = y_possibles[classes, x_c]  
    noise = 0.125 * np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=(N,))

    return 1.75 * np.stack(((x_c-0.5) + noise[:,0], (y_c-0.5) + noise[:,1]), axis=1)


def _get_features_k3(classes, multiplier):
    N = len(classes)

    y_possibles = np.array([[0,1,2], [1,2,0], [2,0,1]])
    x_c = np.random.randint(0, 3, size=(N,))
    y_c = y_possibles[classes, x_c]
    noise = 0.125 * np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=(N,))
    return multiplier * np.stack(((x_c-1) + noise[:,0], (y_c-1) + noise[:,1]), axis=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if True:
        N = 2000
        classes = np.random.randint(0,2, size=(N,))

        X = _get_features_k2(classes)

        plt.figure()
        plt.scatter(X[:,0], X[:,1], c=classes)
        plt.show()
    
    if True:
        N = 2000
        classes = np.random.randint(0,3, size=(N,))
        
        X = _get_features_k3(classes)
        
        plt.figure()
        plt.scatter(X[:,0], X[:,1], c=classes)
        plt.show()