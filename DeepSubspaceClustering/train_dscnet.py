import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import time
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wfutils.infofile
import wfutils.mmap
import wfutils.lrschedule
from nets.dsc import DSCNet
from data.encpipe import AutoencoderDataPipe
import wfutils.log
import wfutils.infofile
import wfutils.lrschedule
from wfutils.progess_printer import ProgressPrinter
from hpc_util.autopath import get_hpc_training_data_path, get_hpc_validation_data_path


@tf.function
def dsc_train_step(x, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(x, y_pred)
        loss_value += tf.reduce_sum(model.losses)  # adding extra losses added during the forward pass
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


@tf.function
def dsc_validation_step(x, model, loss_fn):
    y_pred = model(x)
    loss_value = loss_fn(x, y_pred)
    return loss_value


@tf.function
def reconstruct(x, model):
    return model(x)


parser = argparse.ArgumentParser()
# Data parameters
parser.add_argument('-path_train_data', required=False, type=str, default=r"E:\full32_redist")
parser.add_argument('-path_val_data', required=False, type=str, default=r"E:\validate32_redist")
parser.add_argument("-input_shape", required=False, type=int, nargs="+", default=[64,64,10])

# Training parameters
parser.add_argument("-epochs", required=False, type=int, default=70)
parser.add_argument("-optimizer", required=False, type=str, default='Adam')
parser.add_argument("-lr_start", required=False, type=float, default=0.001)
parser.add_argument("-lr_steps", required=False, type=int, nargs="+", default=[25, 40, 55])
parser.add_argument("-lr_multiplier", required=False, type=float, default=0.1)
parser.add_argument("-momentum", required=False, type=float, default=0.9)
parser.add_argument("-batch_size", required=False, type=int, default=64)

# DSC-Net Parameters
parser.add_argument("-encode_filters", required=False, type=int , nargs="+", default=[16, 32])
parser.add_argument("-max_pool_strides", required=False, type=int, nargs="+", default=[2,2])
parser.add_argument("-weight_decay_l2", required=False, type=float, default=1e-4)
parser.add_argument("-weight_decay_coef_l1", required=False, type=float, default=None)
parser.add_argument("-weight_decay_coef_l2", required=False, type=float, default=1e-4)
parser.add_argument("-constraint_type", required=False, type=str, default="hardset")
parser.add_argument("-alpha", required=False, type=float, default=1.0)


if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()

    # Adjust data path for hpc use
    if args.path_train_data == 'HPC':
        args.path_train_data = get_hpc_training_data_path()
        args.path_val_data = get_hpc_validation_data_path()

    # Create output folder
    path_output_folder = wfutils.log.create_output_folder("DSCNetTraining")
    path_coef_folder = os.path.join(path_output_folder, 'Matrices')
    os.makedirs(path_coef_folder)
    wfutils.log.log_arguments(path_output_folder, args)
    series_log = wfutils.log.SeriesLog(
        path_output_folder=path_output_folder, 
        header_elements=["epoch", "iter", "time", "loss", "val_loss", "lr"],
        filename='series.txt')

    # Loading training and validation data
    train_pipe = AutoencoderDataPipe(path_data=args.path_train_data, batch_size=args.batch_size)
    val_pipe = AutoencoderDataPipe(path_data=args.path_val_data, batch_size=args.batch_size)

    # Create network model
    dscnet = DSCNet(
        input_shape=tuple(args.input_shape),
        encode_filters=args.encode_filters,
        max_pool_strides=args.max_pool_strides,
        l2=args.weight_decay_l2,
        selfexpr_l1=args.weight_decay_coef_l1, 
        selfexpr_l2=args.weight_decay_coef_l2,
        selfexpr_constraint_type=args.constraint_type,
        alpha=args.alpha,
    )
    
    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_start, beta_1=args.momentum)
    elif args.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_start, momentum=args.momentum)
    else:
        raise Exception("Unsupported optimizer: %s" % args.optimizer)

    # Loss function
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Progress printers
    train_printer = ProgressPrinter(steps=train_pipe.number_of_batches, newline_at_end=False)
    val_printer = ProgressPrinter(steps=val_pipe.number_of_batches, header="", print_evolution_number=False)

    # main loop
    t0 = time.time()
    for epoch in range(args.epochs):
        t1 = time.time()

        # training loop
        train_printer.start()
        training_loss = 0
        for step, x in enumerate(train_pipe):
            training_loss += dsc_train_step(x, dscnet, loss_fn, optimizer).numpy()
            train_printer.step()
        training_loss = training_loss / step

        #if(np.isnan(training_loss))

        # validation loop
        val_printer.start()
        validation_loss = 0
        for step, x in enumerate(val_pipe):
            validation_loss += dsc_validation_step(x, dscnet, loss_fn).numpy()
            val_printer.step()
        validation_loss = validation_loss / step

        # print and log progress 
        print("  loss: %.5f val_loss: %.5f time: %d" % (training_loss, validation_loss, round(time.time() - t1)), flush=True)

        series_log.log(elements=[
            epoch + 1,
            (epoch + 1) * train_pipe.number_of_batches,
            time.time() - t0,
            training_loss,
            validation_loss,
            optimizer.learning_rate.numpy()
        ])

        # Changing learning rate
        learning_rate = wfutils.lrschedule.lr_scheduler(epoch + 1, None, args.lr_start, args.lr_steps, args.lr_multiplier)
        optimizer.learning_rate.assign(learning_rate)
    
        # Save coefficient matrix
        path_coef_matrix = os.path.join(path_coef_folder, 'cm_%d.npy' % (epoch + 1))
        np.save(path_coef_matrix, dscnet.get_layer('selfexp').get_weights()[0])

        # Save model    
        dscnet.save(os.path.join(path_output_folder, 'model_end'))


    # Plot Examples
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplot(1, 1 + len(example_reconstructions), 1)
    # plt.imshow(example_batch[0,:,:,0], cmap='gray')

    # for i in range(len(example_reconstructions)):
    #     plt.subplot(1, 1 + len(example_reconstructions), i+2)
    #     plt.imshow(example_reconstructions[i][0,:,:,0], cmap='gray')

    # plt.show()
   

    