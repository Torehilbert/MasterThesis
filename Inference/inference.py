import argparse
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.infofile
import wfutils.mmap


def add_inference_file_to_training(path_training, path_test_data, batch_size=64, override=False, net_selection='best', out_name="test_acc.txt", class_equalize=False):
    path_model = os.path.join(path_training, 'model_'+net_selection)
    path_out_file = os.path.join(path_training, out_name)
    if os.path.isfile(path_out_file) and not override:
        print("Warning: Skipped %s because of existing test accuracy file!")
        return

    acc, class_accs, ns = run_inference_from_paths(path_model, path_test_data, batch_size=batch_size, class_equalize=class_equalize)
    f = open(path_out_file, 'w')
    f.write(str(acc) + " " + " ".join([str(acc) for acc in class_accs]) + "\n")
    f.write(str(sum(ns)) + " " + " ".join([str(n) for n in ns]))
    f.close()


def run_inference_from_paths(path_model, path_data, batch_size=64, class_equalize=False):
    # model
    model = tf.keras.models.load_model(path_model)
    channels_used = find_channels_used(path_model)

    # data
    shapes, data_dtype = wfutils.infofile.read_info(wfutils.infofile.get_path_to_infofile(path_data))
    mmaps = wfutils.mmap.get_class_mmaps_read(path_data, shapes, data_dtype)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    return run_inference(model, data_mmaps=mmaps, loss_fn=loss_fn, channels_used=channels_used, batch_size=batch_size, class_equalize=class_equalize)


def run_inference(model, data_mmaps, loss_fn, batch_size=64, channels_used=None, class_equalize=False):
    mmaps = data_mmaps
    ns = []
    accuracies = []
    for clas, mmap in enumerate(mmaps):
        class_accuracy_sum = 0
        n = 0
        N = mmap.shape[0]

        cursor = 0
        while cursor < N:
            # determine cursor range
            cursor_end = cursor + batch_size
            if cursor_end > N:
                cursor_end = N

            # extract X
            x = mmap[cursor:cursor_end,:,:, channels_used] if channels_used is not None else mmap[cursor:cursor_end]
            cursor = cursor_end

            # create Y
            y = clas * np.ones(shape=(x.shape[0], 1))

            # evaluate and save results
            accuracy = validation_step(x,y,model,loss_fn)
            n += x.shape[0]
            class_accuracy_sum += x.shape[0] * accuracy
        
        # normalize accumulated accuracies
        class_accuracy = class_accuracy_sum / n

        ns.append(n)
        accuracies.append(class_accuracy.numpy())
    
    if not class_equalize:
        total_accuracy = 0
        weights = [n/sum(ns) for n in ns]
        for (w,acc) in zip(weights, accuracies):
            total_accuracy += w*acc
    
        return total_accuracy, accuracies, ns
    else:
        return np.mean(accuracies), accuracies, ns



def find_channels_used(path_to_model):
    path_training = os.path.dirname(path_to_model)
    path_training_args = os.path.join(path_training, 'args.txt')
    
    if not os.path.isfile(path_training_args):
        raise Exception("The file " + path_training_args + " does not exist!")

    f = open(path_training_args, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        if line.startswith('use_channels'):
            val = line.split("=")[1].strip()
            return None if val=='None' else [int(channel_char) for channel_char in val[1:-1].split(", ")]

    
    raise Exception("Could not find argument <use_channels> in args.txt!")


#@tf.function
def validation_step(x, y, model, loss_fn):
    y_pred = model(x)
    loss_value = loss_fn(y, y_pred)
    pred_class = tf.cast(tf.argmax(y_pred, axis=1), dtype=np.float32)
    accuracy = 1 - tf.reduce_mean(tf.abs(tf.sign(pred_class - tf.cast(y, dtype=np.float32))))
    return accuracy


parser = argparse.ArgumentParser()
parser.add_argument('-path_model', required=False, type=str, default=r"D:\Speciale\Code\output\Performance Trainings\C0123456789\C0123456789_Run1\model_best")
parser.add_argument('-path_test_data', required=False, type=str, default=r"E:\test32")
parser.add_argument('-batch_size', required=False, type=int, default=64)


if __name__ == "__main__":
    args = parser.parse_args()

    add_inference_file_to_training(path_training=os.path.dirname(args.path_model), path_test_data=args.path_test_data)

    # acc, acc_class, ns = run_inference_from_paths(path_model=args.path_model, path_data=args.path_test_data)
    # print("Total accuracy: ", acc)
    # print("     Class accuracies: ", acc_class)
    # print("     N: ", ns)
