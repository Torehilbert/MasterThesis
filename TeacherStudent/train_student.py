import os
import sys
import argparse
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nets.student import student_network, student_channel_importance
import wfutils.log
import wfutils.infofile
import wfutils.lrschedule
from plotlib.plot_training_curve import plot_training_curve_from_file
from plotlib.plot_pair_scatter import plot_pair_scatter

parser = argparse.ArgumentParser()
parser.add_argument("-path_data_train", required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Phantom Teacher 16\Run 3\TC Train 16")
parser.add_argument("-path_data_val", required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Phantom Teacher 16\Run 3\TC Val 16")
parser.add_argument("-epochs", required=False, type=int, default=300)
parser.add_argument("-optimizer", required=False, type=str, default='SGD')
parser.add_argument("-lr_start", required=False, type=int, default=0.1)
parser.add_argument("-lr_steps", required=False, type=int, nargs="+", default=[100,200])
parser.add_argument("-batch_size", required=False, type=int, default=32)
parser.add_argument("-lambda_input_reg", required=False, type=float, default=0.01)
parser.add_argument("-weight_decay", required=False, type=float, default=0.0001)
parser.add_argument("-student_hidden_layer_factor", required=False, type=int, default=5)
parser.add_argument("-input_regularization_type", required=False, type=str, default='new')
parser.add_argument("-inp_reg_steps", required=False, type=int, nargs="+", default=None)
parser.add_argument("-inp_reg_multiplier", required=False, type=float, default=1.5)

@tf.function
def train_step(x, y, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        codes_pred = model(x, training=True)
        loss_value = loss_fn(y, codes_pred)
        loss_reg = tf.reduce_sum(model.losses)
        loss_value += loss_reg  # adding extra losses added during the forward pass
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))   
    return loss_value, loss_reg


#@tf.function
def validation_step(x, y, model, loss_fn):
    codes_pred = model(x)
    loss_value = loss_fn(y, codes_pred)
    return loss_value, codes_pred


@tf.function
def calculate_channel_importance_scores(W):
    return tf.reduce_sum(tf.multiply(W, W), axis=[0,1,3])


if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()

    # Load data input info
    shapes, data_dtype = wfutils.infofile.read_info_codes(wfutils.infofile.get_path_to_infofile(args.path_data_train))
    shapes_val, data_dtype_val = wfutils.infofile.read_info_codes(wfutils.infofile.get_path_to_infofile(args.path_data_val))

    # Error checking
    if shapes[0][1] != shapes_val[0][1] or shapes[0][2] != shapes_val[0][2] or shapes[0][3] != shapes_val[0][3]:
        raise Exception("X in training directory and X in validation directory have different shapes other than in the batch dimension!")

    if shapes[1][1] != shapes_val[1][1]:
        raise Exception("Y in training directory and Y in validation directory have different shapes other than in the batch dimension!")

    # Create output folder
    path_output_folder = wfutils.log.create_output_folder("StudentTraining")
    wfutils.log.log_arguments(path_output_folder, args)
    series_log = wfutils.log.SeriesLog(
        path_output_folder=path_output_folder, 
        header_elements=["epoch", "loss", "loss_main", "loss_reg", "val_loss", "lr"],
        filename='series.txt')

    # Create student network
    if args.inp_reg_steps is None:
        student = student_network(
            input_shape=(shapes[0][1], shapes[0][2], shapes[0][3]), 
            n_output_codes=shapes[1][1], 
            hidden_factor=args.student_hidden_layer_factor, 
            l1_input_reg=args.lambda_input_reg, 
            l2_hidden_reg=args.weight_decay,
            input_reg_type=args.input_regularization_type)
    else:
        student, input_regularizer = student_network(
            input_shape=(shapes[0][1], shapes[0][2], shapes[0][3]), 
            n_output_codes=shapes[1][1], 
            hidden_factor=args.student_hidden_layer_factor, 
            l1_input_reg=args.lambda_input_reg, 
            l2_hidden_reg=args.weight_decay,
            input_reg_type=args.input_regularization_type,
            return_input_regularizer=True)   

    # Load data
    Xtrain = np.memmap(os.path.join(args.path_data_train, 'X.npy'), shape=shapes[0], dtype=data_dtype, mode='r')[:]
    Ytrain = np.memmap(os.path.join(args.path_data_train, 'Y.npy'), shape=shapes[1], dtype=data_dtype, mode='r')[:]
    Xval = np.memmap(os.path.join(args.path_data_val, 'X.npy'), shape=shapes_val[0], dtype=data_dtype, mode='r')[:]
    Yval = np.memmap(os.path.join(args.path_data_val, 'Y.npy'), shape=shapes_val[1], dtype=data_dtype, mode='r')[:]

    # Define loss function and optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()

    if args.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_start)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_start)

    # Training loop
    losses_train_main = []
    losses_train_reg = []
    losses_val = []
    importance_scores_by_epoch = []
    
    n_batches = Xtrain.shape[0] // args.batch_size
    n_batches_validation = Xval.shape[0] // args.batch_size
    for epoch in range(args.epochs):  
        # Generate random train data indices
        idx_train = np.linspace(0, Xtrain.shape[0], Xtrain.shape[0], endpoint=False, dtype=np.uint64)
        random.shuffle(idx_train)
        idx_cursor = 0

        training_loss_main = 0
        training_loss_reg = 0
        for batch in range(n_batches):
            # Find indices for batch
            idx = idx_train[idx_cursor:(idx_cursor+args.batch_size)]
            idx_cursor += args.batch_size

            # Train step
            loss, loss_reg = train_step(Xtrain[idx], Ytrain[idx], student, loss_fn, optimizer)

            # Append batch-specific loss to epoch-specific loss lists
            training_loss_main += loss.numpy() - loss_reg.numpy()
            training_loss_reg += loss_reg.numpy()

        # Append training losses
        training_loss_main = training_loss_main / n_batches
        training_loss_reg = training_loss_reg / n_batches
        losses_train_main.append(training_loss_main)
        losses_train_reg.append(training_loss_reg)
        
        # Validate
        idx_cursor = 0
        validation_loss = 0
        for batch in range(n_batches_validation):
            xbatch = Xval[idx_cursor:(idx_cursor+args.batch_size)]
            ybatch = Yval[idx_cursor:(idx_cursor+args.batch_size)]
            idx_cursor += args.batch_size
            loss, _ = validation_step(tf.Variable(xbatch), tf.Variable(ybatch), student, loss_fn)
            validation_loss += loss.numpy()
        validation_loss = validation_loss / n_batches_validation
        losses_val.append(validation_loss)

        # Print status
        msg = "Epoch %4d" % (epoch + 1)
        msg += "  Loss %4.4f (%.4f + %.4f)" % (training_loss_main + training_loss_reg, training_loss_main, training_loss_reg)
        msg += "  Validation loss %4.4f" % validation_loss
        print(msg)

        # Log training progress "epoch", "loss", "loss_main", "loss_reg", "val_loss", "lr"
        series_log.log(elements=[
            epoch + 1,
            training_loss_main + training_loss_reg,
            training_loss_main,
            training_loss_reg,
            validation_loss,
            optimizer.learning_rate.numpy()
        ])

        # Log channel importance scores
        importance_scores = calculate_channel_importance_scores(student.layers[1].get_weights()[0])
        importance_scores_by_epoch.append(importance_scores.numpy())
        
        # Changing learning rate
        learning_rate = wfutils.lrschedule.lr_scheduler(epoch + 1, None, 0.1, args.lr_steps, 0.1)
        optimizer.learning_rate.assign(learning_rate) 

        

    # Plot training series
    plot_training_curve_from_file(
        path_to_file=series_log.path, 
        ykeys=['loss', 'loss_main', 'loss_reg', 'val_loss'],
        color_ids=[0,0,0,1],
        linewidths=[1,0.5,0.5,1],
        linestyles=['solid', 'dashed', 'solid', 'solid'],
        show=False,
        save_path=os.path.join(path_output_folder, 'loss.png')
    )

    # Get codes prediction on entire validation set
    codes_prediction = np.zeros(shape=(n_batches_validation * args.batch_size, Yval.shape[1]), dtype="float32")
    idx_cursor = 0
    for batch in range(n_batches_validation):
        # Manager start and end index
        istart = idx_cursor
        iend = idx_cursor + args.batch_size
        idx_cursor += args.batch_size

        # Validation step
        _, preds = validation_step(Xval[istart:iend], Yval[istart:iend], student, loss_fn)
        codes_prediction[istart:iend,:] = preds

    # Plot codes prediction
    plot_pair_scatter(
        Y1=Yval, 
        Y2=codes_prediction,
        show=False,
        save_path=os.path.join(path_output_folder, 'predictions_versus_true.png'))

    # Compute channel importance and ranking
    weights = student.layers[1].get_weights()[0]
    importance_scores = student_channel_importance(weights)
    importance_scores_normalized = importance_scores / np.max(importance_scores)
    ranking = np.flip(np.argsort(importance_scores))

    # Save student network
    student.save(os.path.join(path_output_folder, 'student'))

    # Save ranking
    (pd.DataFrame({'rank (all)':ranking})).to_csv(os.path.join(path_output_folder, 'ranking.csv'), index=False)

    # Save importance scores
    (pd.DataFrame({'scores':importance_scores, 'scores normalized':importance_scores_normalized})).to_csv(os.path.join(path_output_folder, 'importance_scores.csv'), index=False)

    # Save importance score progression by epoch
    stacked_importance_scores = np.stack(importance_scores_by_epoch)
    np.savetxt(os.path.join(path_output_folder, 'importance_scores_progression.csv'), stacked_importance_scores, delimiter=',')
    print(ranking)
