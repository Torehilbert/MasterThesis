import os


INFO_FILE_NAME = "info.txt"
CHANNEL_ORDER_FILE_NAME = "channel_order.txt"


def get_path_to_infofile(directory):
    return os.path.join(directory, INFO_FILE_NAME)


def get_path_to_channel_order(directory):
    return os.path.join(directory, CHANNEL_ORDER_FILE_NAME)


def read_info(path_info_file):
    shapes = []
    data_dtype = None
    with open(path_info_file, 'r') as f:
        for i in range(3):
            shapes.append(tuple((int(s) for s in f.readline().split(","))))
        data_dtype = f.readline()
    return shapes, data_dtype


def create_info(path_info_file, shapes, data_dtype):
    with open(path_info_file, 'w') as infofile:
        for i in range(len(shapes)):
            num_samples = shapes[i][0]
            num_rows = shapes[i][1]
            num_cols = shapes[i][2]
            num_channels = shapes[i][3]

            infofile.write("%d,%d,%d,%d\n" % (num_samples, num_rows, num_cols, num_channels))
        infofile.write("%s" % data_dtype)


def create_info_codes(path_info_file, shapes, data_dtype):
    with open(path_info_file, 'w') as infofile:   
        infofile.write("%d,%d,%d,%d\n" % (shapes[0][0], shapes[0][1], shapes[0][2], shapes[0][3])) 
        infofile.write("%d,%d\n" % (shapes[1][0], shapes[1][1]))
        infofile.write("%s" % data_dtype)   


def read_info_codes(path_info_file):
    shapes = []
    data_dtype = None
    with open(path_info_file, 'r') as f:
        for _ in range(2):
            shapes.append(tuple((int(s) for s in f.readline().split(","))))
        data_dtype = f.readline()
    return shapes, data_dtype


def create_channel_order(path_order_file, labels):
    with open(path_order_file, 'w') as f:
        for label in labels:
            f.write(label + "\n")


def read_channel_order(path_order_file, exclude_prefix=False):
    channels = []
    with open(path_order_file, 'r') as f:
        channels = [ch.replace("\n", "") for ch in f.readlines()]
    
    if exclude_prefix:
        channels_new = []
        for i,ch in enumerate(channels):
            splits = ch.split("_")
            if len(splits)==2:
                channels_new.append(splits[1])
            else:
                channels_new.append(ch)
        channels = channels_new
    
    return channels
