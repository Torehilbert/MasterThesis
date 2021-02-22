import os
from datetime import datetime
import time
import numpy as np
import argparse
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets.resnet import ResNetV2, ResNetV2Bottleneck
import data.cifar10 as cifar10
from plotlib.learn_curve import plot_learn_curve
import wfutils

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', required=False, type=int, default=2)
parser.add_argument('-batch_size', required=False, type=int, default=64)
parser.add_argument('-lr_start', required=False, type=float, default=0.1)
parser.add_argument('-lr_multiplier', required=False, type=float, default=0.1)
parser.add_argument('-lr_steps', required=False, nargs='+', type=int, default=[40, 60])
parser.add_argument('-momentum', required=False, type=float, default=0.9)
parser.add_argument('-weight_decay', required=False, type=float, default=0.0001)
parser.add_argument('-resnet_block_sizes', required=False, type=int, nargs='+', default=[3,3,3])
parser.add_argument('-resnet_filter_sizes', required=False, type=int, nargs='+', default=[16,32,64])
parser.add_argument('-resnet_size_reductions', required=False, type=int, nargs='+', default=[0, 1, 1])
parser.add_argument('-resnet_stem_params', required=False, type=int, nargs='+', default=[16,3,1])
parser.add_argument('-resnet_stem_max_pool', required=False, type=int, default=0)
parser.add_argument('-resnet_stem_max_pool_size', required=False, type=int, default=3)
parser.add_argument('-resnet_use_bias', required=False, type=int, default=0)
parser.add_argument('-resnet_bottleneck_units', required=False, type=int, default=0)
parser.add_argument('-aug_width_shift', required=False, type=float, default=0.2)
parser.add_argument('-aug_height_shift', required=False, type=float, default=0.2)
parser.add_argument('-aug_horizontal_flip', required=False, type=int, default=1)
parser.add_argument('-image_size', required=False, type=int, default=32)

def lr_scheduler(epoch, lr_current, lr_start, lr_steps, lr_multiplier):   
    multiplier = 1
    for i in range(len(lr_steps)):
        if epoch > lr_steps[i]:
            multiplier *= lr_multiplier

    return multiplier * lr_start


SCRIPT_IDENTIFIER = "ResNet_Cifar10_Reproduction"


if __name__ == "__main__":

    args = parser.parse_args()

    # Create output folder
    path_output_folder = wfutils.create_output_folder(SCRIPT_IDENTIFIER)
    wfutils.log_arguments(path_output_folder, args)

    # A) load data 
    train, test = cifar10.load(resolution=None if args.image_size==32 else (args.image_size, args.image_size))
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=args.aug_width_shift, 
                                                                height_shift_range=args.aug_height_shift, 
                                                                horizontal_flip=bool(args.aug_horizontal_flip))
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator().flow(test[0], test[1], batch_size=args.batch_size)

    # B) create model
    model_delegate = ResNetV2Bottleneck if bool(args.resnet_bottleneck_units) else ResNetV2
    model = model_delegate(input_shape=(None, None, 3), 
                    n_classes=10, 
                    param_stem=[args.resnet_stem_params[0], args.resnet_stem_params[1], args.resnet_stem_params[2]], 
                    ns=args.resnet_block_sizes, 
                    filters=args.resnet_filter_sizes, 
                    reduce_size=[bool(i) for i in args.resnet_size_reductions],
                    use_bias=bool(args.resnet_use_bias),
                    stem_max_pooling=bool(args.resnet_stem_max_pool),
                    stem_max_pool_size=(args.resnet_stem_max_pool_size, args.resnet_stem_max_pool_size),
                    weight_decay=args.weight_decay)
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr_start, momentum=args.momentum), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    # C) train model
    checkpoint_path = os.path.join(path_output_folder, 'cp.ckpt')
    callback_lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr_current : lr_scheduler(epoch, lr_current, args.lr_start, args.lr_steps, args.lr_multiplier))
    callback_model = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_accuracy", save_weights_only=True, verbose=1, save_best_only=True)
    t0 = time.time()
    track = model.fit(train_gen.flow(train[0], train[1], 
                    batch_size=args.batch_size), 
                    steps_per_epoch=len(train[0])/args.batch_size,
                    epochs=args.epochs,
                    validation_data=test_gen, validation_steps=len(test[0]) / args.batch_size,
                    callbacks=[callback_lr, callback_model],
                    verbose=2)
    t1 = time.time()

    # D) create log/results directory
    n_epochs = len(track.history['accuracy'])
    epochs = np.linspace(1, n_epochs, n_epochs)
    iterations = epochs * len(train[0]) / args.batch_size
    wallclock = epochs/len(epochs) * (t1 - t0)

    # raw file txt
    file_raw = open(os.path.join(path_output_folder, 'raw.txt'), 'w')
    file_raw.write("epoch,iter,time,accuracy,val_accuracy\n")
    for i in range(len(iterations)):
        file_raw.write("%f,%f,%f,%f,%f\n" % (epochs[i], iterations[i], wallclock[i], track.history['accuracy'][i], track.history['val_accuracy'][i]))
    file_raw.close()

    # plot
    plot_learn_curve(x=iterations, 
                    acc=np.array(track.history['accuracy']), 
                    val_acc=np.array(track.history['val_accuracy']), 
                    use_error_rate=False, 
                    show=False,
                    save_path=os.path.join(path_output_folder, 'plot.png'))
    
