import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
import numpy as np

PATH_FILE = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\series.txt"


if __name__ == "__main__":
    df = pd.read_csv(PATH_FILE)
    print(df.columns)

    plt.figure(figsize=(6,4))
    plt.plot(df['epoch'], df['val_loss'], color=sns.color_palette()[1], label='Validation', linewidth=2)
    plt.plot(df['epoch'], df['loss'], color=sns.color_palette()[0],  label='Training', linewidth=2)
    plt.plot(df['epoch'], df['loss_x'], color=sns.color_palette()[0], linestyle='dashed', label='Training - X reconstruction', linewidth=1)
    plt.plot(df['epoch'], df['loss_u'], color=sns.color_palette()[0], linestyle='dashdot', label='Training - U reconstruction', linewidth=1)
    plt.plot(df['epoch'], df['loss_reg'], color=sns.color_palette()[0], linestyle='dotted', label='Training - regularization', linewidth=1)

    #plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim([0,0.006])
    plt.legend()
    plt.tight_layout()

    #plt.savefig("dsc_loss_progress.pdf")
    #plt.savefig("dsc_loss_progress.png", dpi=250)
    
    plt.show()
    
