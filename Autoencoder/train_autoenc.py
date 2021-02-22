import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Autoencoder.autoencoder import create_autoencoder
from Autoencoder.autoencoder_channelwise import create_channel_wise_autoencoder
from Autoencoder.dscnet import create_dscnet
from data.encpipe import AutoencoderDataPipe
from Autoencoder.autoenc_trainer import AutoencoderTrainer
from Autoencoder.dsc_trainer import DSCTrainer
from hpc_util.autopath import get_hpc_training_data_path, get_hpc_validation_data_path, get_hpc_scaler_file_path
from wfutils.minmaxscaler import MinMaxScaler


parser = argparse.ArgumentParser()
### DATA
parser.add_argument("-path_train_data", required=False, type=str, default=r"E:\full32_redist_crop_rescale")
parser.add_argument("-path_val_data", required=False, type=str, default=r"E:\validate32_redist_crop_rescale")
parser.add_argument("-path_scaler", required=False, type=str, default=None) #r"E:\normalization_values\quantile_p1_crop.txt")
parser.add_argument("-input_shape", required=False, type=int, nargs="+", default=[32,32,10])
parser.add_argument("-cropping", required=False, type=int, default=0)

### TRAINING
parser.add_argument("-epochs", required=False, type=int, default=400)
parser.add_argument("-batch_size", required=False, type=int, default=32)
parser.add_argument("-improvement_method", required=False, type=str, default="early_stop") # "lr_mult 0.1 2 early_stop" "early_stop"
parser.add_argument("-improvement_patience", required=False, type=int, default=10)

# Optimization
parser.add_argument("-loss", required=False, type=str, default="mse")
parser.add_argument("-optimizer", required=False, type=str, default="Adam")
parser.add_argument("-learning_rate_initial", required=False, type=float, default=0.0001)
parser.add_argument("-momentum", required=False, type=float, default=0.9)
parser.add_argument("-learning_rate_values", required=False, type=float, nargs="+", default=[0.01, 0.001, 0.0001])
parser.add_argument("-learning_rate_steps", required=False, type=int, nargs="+", default=[250, 400])

parser.add_argument("-weight_decay", required=False, type=float, default=1e-5)
parser.add_argument("-weight_decay_selfexp", required=False, type=float, default=1e-5)

# Augmentation
parser.add_argument("-aug_noise_level", required=False, type=float, default=0.00)
parser.add_argument("-aug_random_flip", required=False, type=int, default=0)
parser.add_argument("-aug_random_shift", required=False, type=float, default=0.0)

### ARCHITECTURE
parser.add_argument("-architecture_code", required=False, type=int, default=2) # 0: normal autoencoder, 1: channel wise autoencoder, 2: channel wise dscnet
parser.add_argument("-path_existing_model", required=False, type=str, default=None)
ARCHITECTURE_TAGS = ["AE_Training", "AE_CW_Training", "DSCNet_Training"]

# Encoder
parser.add_argument("-encoder_filters", required=False, type=int, nargs="+", default=[32,64])
parser.add_argument("-encoder_strides", required=False, type=int, nargs="+", default=1)
parser.add_argument("-encoder_kernel_sizes", required=False, type=int, nargs="+", default=3)
parser.add_argument("-encoder_batch_norm", required=False, type=int, nargs="+", default=[0])
parser.add_argument("-encoder_activations", required=False, type=str, nargs="+", default=["relu", "relu"])
parser.add_argument("-encoder_maxpool", required=False, type=int, nargs="+", default=[0])

# Decoder
parser.add_argument("-decoder_filters", required=False, type=int, nargs="+", default=[64,32])
parser.add_argument("-decoder_strides", required=False, type=int, nargs="+", default=1)
parser.add_argument("-decoder_kernel_sizes", required=False, type=int,  nargs="+", default=3)
parser.add_argument("-decoder_batch_norm", required=False, type=int, nargs="+", default=[0])
parser.add_argument("-decoder_activations", required=False, type=str, nargs="+", default="relu")
parser.add_argument("-decoder_output_kernel_size", required=False, type=int, default=3)
parser.add_argument("-decoder_output_activation", required=False, type=str, default='sigmoid')



if __name__ == "__main__":
    args = parser.parse_args()

    args.encoder_batch_norm = [(True if i==1 else False) for i in args.encoder_batch_norm]
    args.encoder_maxpool = [(True if i==1 else False) for i in args.encoder_maxpool]
    args.decoder_batch_norm = [(True if i==1 else False) for i in args.decoder_batch_norm]

    # Adjust data path for hpc use
    if args.path_train_data == 'HPC':
        args.path_train_data = get_hpc_training_data_path(rescaled_version=True)
        args.path_val_data = get_hpc_validation_data_path(rescaled_version=True)
        args.path_scaler = None #get_hpc_scaler_file_path() if args.path_scaler is not None else None

    # Define model
    if args.architecture_code == 0:
        model_master = create_autoencoder(
            input_shape=args.input_shape,
            encoder_filters=args.encoder_filters,
            decoder_filters=args.decoder_filters,
            encoder_kernel_sizes=args.encoder_kernel_sizes,
            decoder_kernel_sizes=args.decoder_kernel_sizes,
            encoder_strides=args.encoder_strides,
            decoder_strides=args.decoder_strides,
            encoder_activations=args.encoder_activations,
            decoder_activations=args.decoder_activations,
            decoder_output_kernel_size=args.decoder_output_kernel_size,
            decoder_output_activation=args.decoder_output_activation,
            weight_decay=args.weight_decay
        )
    elif args.architecture_code == 1:
        model_master = create_channel_wise_autoencoder(
            input_shape=args.input_shape,
            encoder_filters=args.encoder_filters,
            decoder_filters=args.decoder_filters,
            encoder_kernel_sizes=args.encoder_kernel_sizes,
            decoder_kernel_sizes=args.decoder_kernel_sizes,
            encoder_strides=args.encoder_strides,
            decoder_strides=args.decoder_strides,
            encoder_activations=args.encoder_activations,
            decoder_activations=args.decoder_activations,
            decoder_output_kernel_size=args.decoder_output_kernel_size,
            decoder_output_activation=args.decoder_output_activation,
            weight_decay=args.weight_decay
        )
    else:
        model_master = create_dscnet(
            input_shape=args.input_shape,
            encoder_filters=args.encoder_filters,
            decoder_filters=args.decoder_filters,
            encoder_kernel_sizes=args.encoder_kernel_sizes,
            decoder_kernel_sizes=args.decoder_kernel_sizes,
            encoder_strides=args.encoder_strides,
            decoder_strides=args.decoder_strides,
            encoder_maxpool=args.encoder_maxpool,
            encoder_activations=args.encoder_activations,
            decoder_activations=args.decoder_activations,
            encoder_batch_norm=args.encoder_batch_norm,
            decoder_batch_norm=args.decoder_batch_norm,
            decoder_output_kernel_size=args.decoder_output_kernel_size,
            decoder_output_activation=args.decoder_output_activation,
            autoencoder_weight_decay=args.weight_decay,
            selfexpressive_weight_decay=args.weight_decay_selfexp    
        )

        if args.path_existing_model is not None:
            model_master.set_weights(tf.keras.models.load_model(args.path_existing_model).get_weights()) 

    # Load data
    scaler = MinMaxScaler(args.path_scaler)

    train_pipe = AutoencoderDataPipe(
        args.path_train_data, 
        batch_size=args.batch_size, 
        rescale_mins=scaler.vmin, 
        rescale_maxs=scaler.vmax,
        cropping=args.cropping,
        aug_noise_level=args.aug_noise_level,
        aug_random_flip=args.aug_random_flip,
        aug_random_shift=args.aug_random_shift
    )
    val_pipe = AutoencoderDataPipe(
        args.path_val_data, 
        batch_size=args.batch_size, 
        rescale_mins=scaler.vmin, 
        rescale_maxs=scaler.vmax,
        cropping=args.cropping
    )

    # Define trainer
    if args.architecture_code == 0 or args.architecture_code == 1:
        trainer = AutoencoderTrainer(
            model=model_master, 
            data=train_pipe, 
            val_data=val_pipe, 
            loss_fn=args.loss, 
            optimizer=args.optimizer, 
            tag=ARCHITECTURE_TAGS[args.architecture_code], 
            args=args)
    else:
        trainer = DSCTrainer(
            model=model_master, 
            data=train_pipe, 
            val_data=val_pipe, 
            loss_fn=args.loss, 
            optimizer=args.optimizer, 
            tag=ARCHITECTURE_TAGS[args.architecture_code], 
            args=args)
        
    
    trainer.setup_improvement_tracker( 
        callback_no_improve=args.improvement_method,
        callback_improve=None,
        patience=args.improvement_patience,
        val_metric_idx=0,
        improvement_sign=-1
    )

    # Begin training
    trainer.train(args.epochs)     
