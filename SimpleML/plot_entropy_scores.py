import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wfutils import CHANNELS
from plotlib.style import COLORS_METHOD

FILE_PATH = r"D:\Speciale\Code\output\SimpleML\Ranking Entropy - Real (Correct)\entropy_scores.csv"
FILE_PATH = r"D:\Speciale\Code\output\SimpleML\Ranking Entropy - Phantom DPI (New)\entropy_scores.csv"

CHANNELS_PHANTOM = ["DPI", "Discr1", "Discr2", "Discr3", "Blur1", "Blur2", "Blur3", "Noisy1", "Noisy2", "Noisy3", "X1", "X2"]

if __name__ == "__main__":
    entropy_scores = np.squeeze(pd.read_csv(FILE_PATH, header=None).values)
    x = np.arange(len(entropy_scores))
    plt.bar(x, entropy_scores, color=COLORS_METHOD[0])
    plt.xticks(ticks=x, labels=CHANNELS_PHANTOM, rotation=90)
    plt.ylabel("Information Entropy")
    plt.tight_layout()
    plt.savefig("entropy_scores.png", bbox_inches = 'tight', pad_inches=0, dpi=500)
    plt.show()
