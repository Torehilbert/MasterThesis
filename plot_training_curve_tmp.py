import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd

path_1 = r"D:\Speciale\Code\output\Performance Trainings\C0123456789\C0123456789_Run1\series.txt"


if __name__ == "__main__":
    df1 = pd.read_csv(path_1)

    plt.figure(figsize=(6,4))
    plt.plot(df1['epoch'], df1['acc'], label='Training')
    plt.plot(df1['epoch'], df1['val_acc'], label='Validation')
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()