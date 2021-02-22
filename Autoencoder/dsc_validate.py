import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.mmap
import wfutils.log
from wfutils.progess_printer import ProgressPrinter
from wfutils.minmaxscaler import MinMaxScaler

parser =argparse.ArgumentParser()
parser.add_argument("-path_data", required=False, type=str, default=r"E:\validate32_redist_crop_rescale")
parser.add_argument("-path_model", required=False, type=str, default=r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--17-38-57_DSC_3264_relu\model_best")
parser.add_argument("-path_scaler", required=False, type=str, default=None) #r"E:\normalization_values\quantile_p1_crop.txt")


if __name__ == "__main__":
    args = parser.parse_args()
    
    # Output folder
    path_output_folder = wfutils.log.create_output_folder("DSCNet_Inference")
    wfutils.log.log_arguments(path_output_folder, args)


    mmaps = wfutils.mmap.get_class_mmaps_read(args.path_data)
    dscnet = tf.keras.models.load_model(args.path_model)
    loss_fn = tf.keras.losses.MeanSquaredError()

    scaler = MinMaxScaler(args.path_scaler)

    # Calculate all losses
    losses_class_wise = []
    unorms_class_wise = []
    for i, mmap in enumerate(mmaps):
        N = mmap.shape[0]
        class_loss = np.zeros(shape=(N, 10, 3), dtype="float32")
        unorms = np.zeros(shape=(N,10,4), dtype="float32")
        printer = ProgressPrinter(N, header='Class %d' % (i+1), sign_start=" |", sign_end="|", print_evolution_number=False)
        printer.start()
        for j in range(N):
            printer.step()
            if(args.path_scaler is not None):
                X = scaler.scale(np.expand_dims(mmap[j, 16:48, 16:48], axis=0))
            else:
                X = np.expand_dims(mmap[j,:,:,:], axis=0)
            Xre, U, Ure = dscnet(X)

            for ch in range(10):
                loss_x = loss_fn(X[0,:,:,ch], Xre[0,:,:,ch]).numpy()
                loss_u = loss_fn(U[0,:,:,:,ch], Ure[0,:,:,:,ch]).numpy()
                loss = loss_x + loss_u

                class_loss[j, ch, 0] = loss_x + loss_u
                class_loss[j, ch, 1] = loss_x
                class_loss[j, ch, 2] = loss_u 

                unorms[j, ch, 0] = np.max(np.abs(U[0,:,:,:,ch]))
                unorms[j, ch, 1] = np.sqrt(np.sum(np.square(U[0,:,:,:,ch])))
                unorms[j, ch, 2] = np.max(np.abs(Ure[0,:,:,:,ch]))
                unorms[j, ch, 3] = np.sqrt(np.sum(np.square(Ure[0,:,:,:,ch])))

        losses_class_wise.append(class_loss)
        unorms_class_wise.append(unorms)

    # Output raw losses
    import pandas as pd
    for c in range(3):
        data_raw = {}
        N = losses_class_wise[c].shape[0]
        channels = losses_class_wise[c].shape[1]
        loss_stats = losses_class_wise[c].shape[2]

        for l in range(loss_stats):
            for ch in range(channels):
                label = "%s %s" % (str(ch), str(l))
                data_raw[label] = losses_class_wise[c][:,ch,l]

        pd.DataFrame(data_raw).to_csv(os.path.join(path_output_folder, 'raw_losses_class_%d.csv' % (c+1)), index=False)

    # Output raw unorms
    for c in range(3):
        data_raw = {}
        N = unorms_class_wise[c].shape[0]
        channels = unorms_class_wise[c].shape[1]
        norm_stats = unorms_class_wise[c].shape[2]

        for s in range(norm_stats):
            for ch in range(channels):
                label = "%s %s" % (str(ch), str(s))
                data_raw[label] = unorms_class_wise[c][:,ch,s]

        pd.DataFrame(data_raw).to_csv(os.path.join(path_output_folder, 'raw_unorms_class_%d.csv' % (c+1)), index=False)        

    # Output C matrix
    C = dscnet.get_layer('selfexp').get_weights()[0]
    np.savetxt(fname=os.path.join(path_output_folder, 'C.csv'), X=C, delimiter=',')
