import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
#parser.add_argument("-path_file", required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\Original Regularization\Teacher A\Run 1\series.txt")
parser.add_argument("-path_file", required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\New Regularization\Teacher B\Run 1\series.txt")


CHANNELS = ["Aperture", "ApodizedAP", "BrightField", "DarkField", "DFIOpen", "DFIPhase", "DPI", "iSSC", "Phase", "UVPhase"]


if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.path_file)

    
    plt.figure(figsize=(6,3))
    plt.plot(df['epoch'], df['val_loss'], label='Validation loss', color=sns.color_palette()[1], linewidth=2)
    plt.plot(df['epoch'], df['loss'], label='Training loss', color=sns.color_palette()[0], linewidth=2)
    plt.plot(df['epoch'], df['loss_main'], label='Reconstruction part of training', color=sns.color_palette()[0], linewidth=1, linestyle="dashed")
    plt.plot(df['epoch'], df['loss_reg'], label='Regularization part of training', color=sns.color_palette()[0], linewidth=1, linestyle="dotted")

    plt.yscale('log')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("teacher_student_training_loss_new.pdf")
    plt.savefig("teacher_student_training_loss_new.png", dpi=250)
