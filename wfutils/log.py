import os
from datetime import datetime
import numpy as np


def get_timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d--%H-%M-%S")


def get_x_series(epochs, batch_per_epoch=None, total_time_elapsed=None):
    if batch_per_epoch is None and total_time_elapsed is None:
        raise Exception("ERROR: <batch_per_epoch> and <total_time_elapsed> cannot both be <None>")
    
    epoch_list = np.linspace(1, epochs, epochs)
    if batch_per_epoch is None:
        return epoch_list, total_time_elapsed * (epoch_list/epochs)
    elif total_time_elapsed is None:
        return epoch_list, epoch_list * batch_per_epoch
    else:
        return epoch_list, epoch_list * batch_per_epoch, total_time_elapsed * (epoch_list/epochs)


def create_output_folder(script_identifier):
    timestamp = get_timestamp()
    output_folder_name = timestamp + "_" + script_identifier
    path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_output_folder = os.path.join(path_root,'output', output_folder_name)
    os.makedirs(path_output_folder)
    return path_output_folder


def log_arguments(path_output_folder, args, filename='args.txt'):
    if args is not None:
        log = open(os.path.join(path_output_folder, filename), 'w')
        for arg in vars(args):
            log.write(str(arg)+"="+str(getattr(args, arg))+"\n")
        log.close()


def log_training_series(path_output_folder, history, epochs=None, iters=None, time=None, filename='raw.txt'):
    N = None

    f = open(os.path.join(path_output_folder, filename), 'w')
    header = ""
    header += "epoch," if epochs is not None else ""
    header += "iter," if iters is not None else ""
    header += "time," if time is not None else ""
    for key in history.keys():
        header += key + ","
        N = len(history[key]) if N is None else N
    f.write(header + "\n")

    for i in range(N):
        line = ""
        if epochs is not None:
            line += "%f," % epochs[i]
        if iters is not None:
            line += "%f," % iters[i]
        if time is not None:
            line += "%f," % time[i]
        for key in history.keys():
            line += "%f," % (history[key][i])
        f.write(line + "\n")
        
    f.close()


class SeriesLog():
    def __init__(self, path_output_folder, header_elements=None, filename='series.txt'):
        self.path = os.path.join(path_output_folder, filename)
        self.num_columns = len(header_elements)
        
        self.batches_per_epoch = None

        # write header
        with open(self.path, "w") as f:
            f.write(','.join(header_elements) + "\n")
    
    def log(self, elements):
        with open(self.path, "a+") as f:
            f.write(','.join([str(el) for el in elements]) + "\n")

    def set_epoch_to_its_constant(self, batches_per_epoch):
        self.batches_per_epoch = batches_per_epoch

    def get_its(self, epoch, batches_per_epoch=None):
        if batches_per_epoch is None:
            batches_per_epoch = self.batches_per_epoch
        
        if batches_per_epoch is None:
            raise Exception("ERROR: \"batches_per_epoch\" is never supplied")
            
        return epoch * batches_per_epoch
