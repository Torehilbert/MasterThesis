import tensorflow as tf
import os
import time
import numpy as np
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nets.resnet as rnets
from plotlib.learn_curve import plot_learn_curve
import data.nppipe as nppipe
import data.pipe as pipe
import wfutils.log
import wfutils.lrschedule
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-epochs', required=False, type=int, default=2)
parser.add_argument('-batch_size', required=False, type=int, default=128)
parser.add_argument('-lr_start', required=False, type=float, default=0.1)
parser.add_argument('-lr_multiplier', required=False, type=float, default=0.1)
parser.add_argument('-lr_steps', required=False, nargs='+', type=int, default=[10, 30, 50])
parser.add_argument('-momentum', required=False, type=float, default=0.9)
parser.add_argument('-weight_decay', required=False, type=float, default=0.0001)
parser.add_argument('-resnet_block_sizes', required=False, type=int, nargs='+', default=[3,5,3])
parser.add_argument('-resnet_filter_sizes', required=False, type=int, nargs='+', default=[32,64,128])
parser.add_argument('-resnet_size_reductions', required=False, type=int, nargs='+', default=[0, 1, 1])
parser.add_argument('-resnet_stem_params', required=False, type=int, nargs='+', default=[64,3,1])
parser.add_argument('-resnet_stem_max_pool', required=False, type=int, default=0)
parser.add_argument('-resnet_stem_max_pool_size', required=False, type=int, default=3)
parser.add_argument('-resnet_use_bias', required=False, type=int, default=0)
parser.add_argument('-resnet_bottleneck_units', required=False, type=int, default=0)
parser.add_argument('-aug_width_shift', required=False, type=float, default=0.3)
parser.add_argument('-aug_height_shift', required=False, type=float, default=0.3)
parser.add_argument('-aug_horizontal_flip', required=False, type=int, default=1)
parser.add_argument('-image_size', required=False, type=int, default=64)
parser.add_argument('-resnet_n_channels', required=False, type=int, default=10)
parser.add_argument('-resnet_n_classes', required=False, type=int, default=3)


if __name__ == "__main__":
    args = parser.parse_args()

    # Create output folder
    path_output_folder = wfutils.log.create_output_folder("Training")
    wfutils.log.log_arguments(path_output_folder, args)

    # Loading training and validation data
    directory = r"D:\Speciale\Data\Real\Validation (split-by-class)\split_Phase"
    
    # X, y = nppipe.load_and_merge_RAM(directory)
    # np.save('Xtrain.npy', X)
    # np.save('Ytrain.npy', y)

    path_x_data = r"C:\Users\ToreH\Desktop\Xtrain.npy"
    path_y_data = r"C:\Users\ToreH\Desktop\Ytrain.npy"
    path_x_data = "Xtrain.npy"
    path_y_data = "Ytrain.npy"


    X = np.load(path_x_data)
    y = np.load(path_y_data)

    VAL_SPLIT = 0.1
    N = X.shape[0]
    N_val = round(VAL_SPLIT * N)
    N_train = round((1-VAL_SPLIT) * N)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=args.aug_width_shift, 
        height_shift_range=args.aug_height_shift, 
        horizontal_flip=bool(args.aug_horizontal_flip), 
        validation_split=VAL_SPLIT)
    
    train_generator = datagen.flow(X,y, batch_size=args.batch_size, subset='training')
    validation_generator = datagen.flow(X,y, batch_size=args.batch_size, subset='validation')

    # Creating network model
    model = rnets.ResNetV2(
        input_shape=(args.image_size, args.image_size, args.resnet_n_channels), 
        n_classes=args.resnet_n_classes, 
        param_stem=args.resnet_stem_params,
        ns=args.resnet_block_sizes,
        filters=args.resnet_filter_sizes,
        reduce_size=[bool(i) for i in args.resnet_size_reductions],
        use_bias=bool(args.resnet_use_bias),
        stem_max_pooling=bool(args.resnet_stem_max_pool),
        stem_max_pool_size=(args.resnet_stem_max_pool_size, args.resnet_stem_max_pool_size),
        weight_decay=args.weight_decay)
        
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr_start, momentum=args.momentum), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy'])

    # Creating callbacks
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr_current : wfutils.lrschedule.lr_scheduler(epoch, lr_current, args.lr_start, args.lr_steps, args.lr_multiplier)),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(path_output_folder, 'cp.ckpt'), 
            monitor="val_accuracy", 
            save_weights_only=True, 
            verbose=1, 
            save_best_only=True)
    ]

    # Start training
    t0 = time.time()
    track = model.fit(
        train_generator, 
        steps_per_epoch=N_train//args.batch_size, 
        epochs=args.epochs, 
        callbacks=callbacks, 
        validation_data=validation_generator, 
        validation_steps=N_val//args.batch_size)
    time_elapsed = time.time() - t0

    # Log training series and construct plots
    x_list_epochs, x_list_iter, x_list_time = wfutils.log.get_x_series(args.epochs, N_train//args.batch_size, time_elapsed)

    wfutils.log.log_training_series(
        path_output_folder=path_output_folder, 
        history=track.history, 
        epochs=x_list_epochs, 
        iters=x_list_iter, 
        time=x_list_time, 
        filename='raw.txt')


    plot_learn_curve(x=x_list_iter, 
                    acc=np.array(track.history['accuracy']), 
                    val_acc=np.array(track.history['val_accuracy']), 
                    use_error_rate=False,
                    show=True)
