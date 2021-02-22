import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.style import COLORS, COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE

USE_INFO_FILE = False
PATH_INFO_FILE = r"E:\test32\info.txt"
MANUAL_NS = [8654,4231,1677]  # [77898, 38089, 15102]
FILE_NAME = "distribution_validation_data.png"
TITLE_BASE = "Validation Data"

def func(x):
    return "%2d%%" % x


def funcDummy(x):
    return "? %"


if __name__ == "__main__":
    # Load info file
    if USE_INFO_FILE:
        ns = []
        with open(PATH_INFO_FILE, 'r') as f:
            for i in range(3):
                ns.append((int(f.readline().split(",")[0])))
    else:
        ns = MANUAL_NS
    

    plt.figure(figsize=(4,4))
    wedges, texts, autotexts = plt.pie(ns, explode=[0.02, 0.03, 0.04], colors=COLORS, autopct=func, textprops=dict(color="w"))
    plt.legend(['1 (Healthy)', '2 (Apoptosis)', '3 (Dead)'])
    plt.setp(autotexts, size=11, weight="bold")

    plt.title("%s (n=%d)" % (TITLE_BASE, sum(ns)))
    #plt.title("Test Data (n=?)")
    plt.savefig(FILE_NAME, bbox_inches = 'tight', pad_inches=0)

    plt.show()
    
    