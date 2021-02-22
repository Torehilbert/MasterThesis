import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def rename_columns_and_drop_redundant_size(df):
    modes = ['Aperture', 'ApodizedAP', 'BrightField', 'DarkField', 'DFIOpen', 'DFIPhase', 'DPI', 'iSSC', 'Phase', 'UVPhase']

    # Rename columns
    vartypes = ["mean", "std", "n"]
    vartypes_new = ["$\mu_{cell}$", "$\sigma_{cell}$", "$size_{cell}$"]
    rename_dict = {}
    for mode in modes:
        for varname, varname_new in zip(vartypes, vartypes_new):
            rename_dict['%s (images_%s)' % (varname, mode)] = "%s (%s)" % (varname_new, mode)
    df = df.rename(columns=rename_dict)

    # Drop redundant size columns
    columns_to_drop = ["%s (%s)" % (vartypes_new[2], mode) for mode in modes]
    del columns_to_drop[0]
    df = df.drop(columns=columns_to_drop)
    df = df.rename(columns={('%s (%s)' % (vartypes_new[2], modes[0])):vartypes_new[2]})  
    return df


PATH_TRAIN_DATA = r"D:\Speciale\Repos\cell crop phantom\output\ExploreSimpleStats\s_13932\data.csv"


if __name__ == "__main__":
    modes = ['Aperture', 'ApodizedAP', 'BrightField', 'DarkField', 'DFIOpen', 'DFIPhase', 'DPI', 'iSSC', 'Phase', 'UVPhase']
    df = pd.read_csv(PATH_TRAIN_DATA, header='infer')
    df = rename_columns_and_drop_redundant_size(df)

    columns_to_test = [1,3,4,6,8,10,12,14,16,18,20]
    idx_class_1 = np.where(df.values[:,0]==1)[0]
    idx_class_3 = np.where(df.values[:,0]==3)[0]

    for i, column in enumerate(columns_to_test):
        tstat, pval = ttest_ind(df.values[idx_class_1, column], df.values[idx_class_3, column])
        print("%s  tstat: %f  pval: %.9f" % (df.columns[column], tstat, pval))
