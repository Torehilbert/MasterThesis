import os


def get_hpc_training_data_paths():
    return [os.path.join(get_hpc_training_data_path(), "%d.npy" % i) for i in [1,2,3]]


def get_hpc_validation_data_paths():
    return [os.path.join(get_hpc_validation_data_path(), "%d.npy" % i) for i in [1,2,3]]


def get_hpc_training_data_path(rescaled_version=False):
    hpc_folder = os.path.dirname(os.path.abspath(__file__))
    code_folder = os.path.dirname(hpc_folder)
    speciale_folder = os.path.dirname(code_folder)
    data_folder = os.path.join(speciale_folder, 'Data')

    if rescaled_version:
        return r"/work1/s144328/full32_redist_crop_rescale"
    else:
        return os.path.join(data_folder, "full32_redist")


def get_hpc_validation_data_path(rescaled_version=False):
    hpc_folder = os.path.dirname(os.path.abspath(__file__))
    code_folder = os.path.dirname(hpc_folder)
    speciale_folder = os.path.dirname(code_folder)
    data_folder = os.path.join(speciale_folder, 'Data')

    if rescaled_version:
        return r"/work1/s144328/validate32_redist_crop_rescale"
    else:
        return os.path.join(data_folder, "validate32_redist")


def get_hpc_scaler_file_path():
    hpc_folder = os.path.dirname(os.path.abspath(__file__))
    code_folder = os.path.dirname(hpc_folder)
    speciale_folder = os.path.dirname(code_folder)
    data_folder = os.path.join(speciale_folder, 'Data')
    norm_folder = os.path.join(data_folder, 'normalization_values')
    return os.path.join(norm_folder, "quantile_p1_crop.txt")


if __name__ == "__main__":
    
    ptrain = get_hpc_training_data_path()
    pval = get_hpc_validation_data_path()

    original_file_path = r"E:\normalization_values\quantile_p1_crop.txt"
    pscale = get_hpc_scaler_file_path()
    print(os.path.basename(original_file_path))
    print(ptrain)
    print(pval)
    print(pscale)