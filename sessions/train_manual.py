import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import time
import numpy as np
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nets.resnet as rnets
import plotlib.learn_curve as plc
import data.pipe as pipe
import data.threadedpipe as threadedpipe
import wfutils.log
import wfutils.lrschedule
import argparse
#import gc

parser = argparse.ArgumentParser()
parser.add_argument('-path_training_data', required=False, type=str, nargs='+', default=[r"E:\full32_redist\1.npy", r"E:\full32_redist\2.npy", r"E:\full32_redist\3.npy"])
parser.add_argument('-path_validation_data', required=False, type=str, nargs='+', default=[r"E:\validate32_redist\1.npy", r"E:\validate32_redist\2.npy", r"E:\validate32_redist\3.npy"])
parser.add_argument('-epochs', required=False, type=int, default=1)
parser.add_argument('-batch_size', required=False, type=int, default=64)
parser.add_argument('-lr_start', required=False, type=float, default=0.1)
parser.add_argument('-lr_multiplier', required=False, type=float, default=0.1)
parser.add_argument('-lr_steps', required=False, nargs='+', type=int, default=[20, 40, 60, 80])
parser.add_argument('-momentum', required=False, type=float, default=0.9)
parser.add_argument('-weight_decay_l1', required=False, type=float, default=0)
parser.add_argument('-weight_decay_l2', required=False, type=float, default=0)
parser.add_argument('-weight_decay_l1_stem', required=False, type=float, default=0)
parser.add_argument('-weight_decay_l2_stem', required=False, type=float, default=0)
parser.add_argument('-resnet_block_sizes', required=False, type=int, nargs='+', default=[3,5,3])
parser.add_argument('-resnet_filter_sizes', required=False, type=int, nargs='+', default=[32,64,128])
parser.add_argument('-resnet_size_reductions', required=False, type=int, nargs='+', default=[0, 1, 1])
parser.add_argument('-resnet_stem_params', required=False, type=int, nargs='+', default=[64,3,1])
parser.add_argument('-resnet_stem_max_pool', required=False, type=int, default=0)
parser.add_argument('-resnet_stem_max_pool_size', required=False, type=int, default=3)
parser.add_argument('-resnet_use_bias', required=False, type=int, default=0)
parser.add_argument('-resnet_bottleneck_units', required=False, type=int, default=0)
parser.add_argument('-aug_translation', required=False, type=float, default=0.2)
parser.add_argument('-aug_horizontal_flip', required=False, type=int, default=1)
parser.add_argument('-aug_noise', required=False, type=float, default=1.0)
parser.add_argument('-image_size', required=False, type=int, default=64)
parser.add_argument('-resnet_n_channels', required=False, type=int, default=5)
parser.add_argument('-resnet_n_classes', required=False, type=int, default=3)
parser.add_argument('-validation_frequency', required=False, type=int, default=1)
parser.add_argument('-optimizer', required=False, type=str, default='SGD')  # 'SGD' or 'Adam'
parser.add_argument('-use_channels', required=False, type=int, nargs="+", default=[1, 3, 4, 6, 7])
parser.add_argument('-save_model_every_epoch', required=False, type=int, default=1)
parser.add_argument('-save_model_epochs', required=False, type=int, nargs="+", default=[])


@tf.function
def train_step(x, y, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(y, y_pred)
        loss_value += tf.reduce_sum(model.losses)  # adding extra losses added during the forward pass
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    pred_class = tf.cast(tf.argmax(y_pred, axis=1), dtype=np.float32)
    accuracy = 1 - tf.reduce_mean(tf.abs(tf.sign(pred_class - tf.cast(y, dtype=np.float32))))
    return loss_value, accuracy

@tf.function
def validation_step(x, y, model, loss_fn):
    y_pred = model(x)
    loss_value = loss_fn(y, y_pred)
    pred_class = tf.cast(tf.argmax(y_pred, axis=1), dtype=np.float32)
    accuracy = 1 - tf.reduce_mean(tf.abs(tf.sign(pred_class - tf.cast(y, dtype=np.float32))))
    return loss_value, accuracy


if __name__ == "__main__":
    args = parser.parse_args() 

    # Create output folder
    folder_postfix = "Training_%d" % random.randint(0, 65000)
    if args.use_channels is not None:
        folder_postfix += "_" + "_".join([str(ch) for ch in args.use_channels]) 
    path_output_folder = wfutils.log.create_output_folder(folder_postfix)
    wfutils.log.log_arguments(path_output_folder, args)
    series_log = wfutils.log.SeriesLog(
        path_output_folder=path_output_folder, 
        header_elements=["epoch", "iter", "time", "acc", "val_acc", "loss", "val_loss", "lr"],
        filename='series.txt')

    # HPC path compensation
    if(args.path_training_data[0] == "HPC"):
        import hpc_util.autopath as hpcap
        hpc_dtrain = hpcap.get_hpc_training_data_path()
        hpc_dval = hpcap.get_hpc_validation_data_path()
        args.path_training_data = [os.path.join(hpc_dtrain, "1.npy"), os.path.join(hpc_dtrain, "2.npy"), os.path.join(hpc_dtrain, "3.npy")]
        args.path_validation_data = [os.path.join(hpc_dval, "1.npy"), os.path.join(hpc_dval, "2.npy"), os.path.join(hpc_dval, "3.npy")]

    # Loading training and validation data
    training_files = args.path_training_data
    validation_files = args.path_validation_data
    
    
    

    datapipe = threadedpipe.ThreadedPipeline(
        path_classes=training_files,
        batch_size=args.batch_size,
        aug_translation=args.aug_translation,
        aug_rotation=None,
        aug_horizontal_flip=bool(args.aug_horizontal_flip),
        aug_noise=args.aug_noise if args.aug_noise > 0.01 else None,
        use_channels='all' if args.use_channels is None else args.use_channels)

    valpipe = pipe.MyDataPipeline(path_classes=validation_files, batch_size=args.batch_size, use_channels='all' if args.use_channels is None else args.use_channels)

    # Creating network model
    model = rnets.ResNetV2(
        input_shape=(args.image_size, args.image_size, args.resnet_n_channels if args.use_channels is None else len(args.use_channels)), 
        n_classes=args.resnet_n_classes, 
        param_stem=args.resnet_stem_params,
        ns=args.resnet_block_sizes,
        filters=args.resnet_filter_sizes,
        reduce_size=[bool(i) for i in args.resnet_size_reductions],
        use_bias=bool(args.resnet_use_bias),
        stem_max_pooling=bool(args.resnet_stem_max_pool),
        stem_max_pool_size=(args.resnet_stem_max_pool_size, args.resnet_stem_max_pool_size),
        weight_decay=(args.weight_decay_l1_stem, args.weight_decay_l2_stem, args.weight_decay_l1, args.weight_decay_l2))
    
    print(model.summary())
    if args.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_start, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_start, beta_1=args.momentum)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    TIME_BEGIN = time.time()
    best_validation_accuracy = -np.inf
    n_its = 0
    for epoch in range(args.epochs):
        print("Epoch %d" % (epoch+1))
        T_GRAD = 0
        T_BATCH_FETCH = 0
        T_VAL = 0
        training_losses = []
        training_accuracies = []
        t0 = time.time()
        t1 = t0
        for step, (x_batch_train, y_batch_train) in enumerate(datapipe):
            t2 = time.time()
            T_BATCH_FETCH += t2 - t1
            loss_part, train_accuracy_part = train_step(x_batch_train, y_batch_train, model, loss_fn, optimizer)
            n_its += 1
            training_losses.append(loss_part)
            training_accuracies.append(train_accuracy_part)
                        
            T_GRAD += time.time() - t2
            t1 = time.time()
            
        
        training_loss = tf.reduce_mean(training_losses)
        training_accuracy = tf.reduce_mean(training_accuracies)

        # Save model every epoch
        if bool(args.save_model_every_epoch):
            model.save(os.path.join(path_output_folder, 'model_end'))

        if epoch in args.save_model_epochs:
            model.save(os.path.join(path_output_folder, "model_%03d" % epoch))

        # Validate on test set
        flag_time_for_validation = (epoch % args.validation_frequency == 0) or (epoch==(args.epochs-1))
        validate_this_epoch = (valpipe is not None) and flag_time_for_validation

        validation_loss = None
        validation_accuracy = None
        if validate_this_epoch:
            # validate
            validation_losses = []
            validation_accuracies = []
            for step, (x_batch_val, y_batch_val) in enumerate(valpipe):
                loss_part, accuracy_part = validation_step(x_batch_val, y_batch_val, model, loss_fn)
                validation_losses.append(loss_part)
                validation_accuracies.append(accuracy_part)

            validation_loss = tf.reduce_mean(validation_losses)
            validation_accuracy = tf.reduce_mean(validation_accuracies)
            T_VAL = time.time() - t1

            # save model if best
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                model.save(os.path.join(path_output_folder, 'model_best'))                
                print("  saved weights with validation accuracy = %.2f" % best_validation_accuracy)

            # print status in console
            print("  loss: %f  accuracy: %f  val_loss: %f  val_accuracy: %f  Time: %f" % (training_loss, training_accuracy, validation_loss, validation_accuracy, time.time() - t0))         
        else:
            if valpipe is None:
                model.save(os.path.join(path_output_folder, 'model'))
            print("  loss: %f  accuracy: %f  val_loss: None  val_accuracy: None  Time: %f" % (training_loss, training_accuracy, time.time() - t0))

        # Saving to training series log
        series_log.log(elements=[
            epoch + 1,
            n_its,
            time.time() - TIME_BEGIN,
            training_accuracy.numpy(),
            validation_accuracy.numpy() if validation_accuracy is not None else np.nan,
            training_loss.numpy(),
            validation_loss.numpy() if validation_loss is not None else np.nan,
            optimizer.learning_rate.numpy()
        ])

        # Changing learning rate
        learning_rate = wfutils.lrschedule.lr_scheduler(epoch + 1, None, args.lr_start, args.lr_steps, args.lr_multiplier)
        optimizer.learning_rate.assign(learning_rate) 

        #gc.collect()
        #print("  dev: computation times (batch fetch: %f,  gradient step: %f,  validation_step: %f)" % (T_BATCH_FETCH, T_GRAD, T_VAL))      


    # Save plot at the end of training
    plc.plot_accuracy_curve_from_file(
        path_to_file=series_log.path,
        show=False,
        save=True,
        filename='accuracy.png'
    )
