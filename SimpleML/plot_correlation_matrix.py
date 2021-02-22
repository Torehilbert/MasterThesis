import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wfutils import CHANNELS
from plotlib.style import COLORS_METHOD

FILE_PATH = r"D:\Speciale\Code\output\SimpleML\Ranking Correlation - Real (Correct)\correlations.csv"

FILE_PATH = r"D:\Speciale\Code\output\SimpleML\Ranking Correlation - Phantom DPI (New)\correlations.csv"
CHANNELS_PHANTOM = ["DPI", "Discr1", "Discr2", "Discr3", "Blur1", "Blur2", "Blur3", "Noisy1", "Noisy2", "Noisy3", "X1", "X2"]


if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH)
    corrm = df.iloc[:,1:].values

    sns.heatmap(corrm, annot=True, fmt=".2f", cmap='RdGy', vmin=-1, vmax=1, annot_kws={"fontsize":8})
    plt.xticks(ticks=np.arange(corrm.shape[0]) + 0.5, labels=CHANNELS_PHANTOM, rotation=90)
    plt.yticks(ticks=np.arange(corrm.shape[0]) + 0.5, labels=CHANNELS_PHANTOM, rotation=0)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", bbox_inches = 'tight', pad_inches=0, dpi=500)
    plt.show()
