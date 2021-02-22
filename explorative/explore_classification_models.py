import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

PATH_TRAIN_DATA = r"D:\Speciale\Repos\cell crop phantom\output\ExploreSimpleStats\s_13932\data.csv"
PATH_VALIDATION_DATA = r"D:\Speciale\Repos\cell crop phantom\output\ExploreSimpleStats\s_2847_validation\data.csv"


def get_accuracy(model, X, y):
    preds = model.predict(X)
    return 1 - np.mean(np.abs(np.sign(preds - y)))


def get_confusion_matrix(model, X, y, normalize=False):
    preds = model.predict(X)
    confmatrix = confusion_matrix(y, preds)
    if normalize:
        confmatrix = confmatrix/np.sum(confmatrix, axis=1)
    return confmatrix


def print_confusion_matrix(confmatrix):
    rows = confmatrix.shape[0]
    for r in range(rows):
        num_strings = []
        for c in range(confmatrix.shape[1]):
            num_strings.append("%5.1f%%" % (100*confmatrix[r,c]))
        print("    [" + ",".join(num_strings) + "]")


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


def print_stats(model, Xtrain, ytrain, Xval, yval, title=None):
    if title is not None:
        print(title)
    print("  Confusion Matrix Training:")
    print_confusion_matrix(get_confusion_matrix(model ,Xtrain, ytrain, normalize=True))
    print("  Confusion Matrix Validation:")
    print_confusion_matrix(get_confusion_matrix(model ,Xval, yval, normalize=True))
    print("  Training accuracy: %.1f%%" % (100*get_accuracy(model, Xtrain, ytrain)))
    print("  Validation accuracy: %.1f%%" % (100*get_accuracy(model, Xval, yval)))


if __name__ == "__main__":    
    df = pd.read_csv(PATH_TRAIN_DATA)
    df = rename_columns_and_drop_redundant_size(df)

    df_val = pd.read_csv(PATH_VALIDATION_DATA)
    df_val = rename_columns_and_drop_redundant_size(df_val)

    Xtrain = df.values[:, 1:]
    ytrain = df.values[:, 0]
    Xval = df_val.values[:,1:]
    yval = df_val.values[:,0]

    # Logistic regression model
    model = LogisticRegression(multi_class='ovr', solver='newton-cg', max_iter=1000)
    model.fit(Xtrain, ytrain)
    print_stats(model, Xtrain, ytrain, Xval, yval, title='Logistic Regression')
    
    # K-nearest neighbors
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(df.values[:,1:], df.values[:,0])
    print_stats(model, Xtrain, ytrain, Xval, yval, title='k-Nearest Neighbours')
