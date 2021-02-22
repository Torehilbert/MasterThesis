import tensorflow as tf
import numpy as np

PATH_MODEL = r"D:\Speciale\Code\output\Performance Trainings\C13467\C13467_Run1\model_best"

if __name__ == "__main__":
    model = tf.keras.models.load_model(PATH_MODEL)
    
    X = np.zeros(shape=(1,64,64,5), dtype=float)
    Y = model(X)
    print(Y)

    model.summary()