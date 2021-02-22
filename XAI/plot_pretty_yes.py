import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
import numpy as np

PATH_FILE = r"D:\Speciale\Code\output\ChannelOcclusion\Class Average\Run1_new\raw_stats.csv"
COLORS = ['royalblue', 'limegreen', 'indianred', 'orange', 'red', 'yellow', 'blue', 'blue', 'blue', 'blue', 'blue']

if __name__ == "__main__":
    df = pd.read_csv(PATH_FILE)
    df['sigma'] = df['dprob standard error'] * np.sqrt(1500)

    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    dfsub = df.loc[df['class']=='all']
    plt.bar(np.arange(10), -dfsub['dprob mean'], color='gray')
    plt.xticks(np.arange(10), dfsub['channel'], rotation=-65)
    plt.xlabel("(a) Aggregated")
    plt.ylabel("Percentage drop")
    plt.subplot(1,2,2)
    for i, class_idx in enumerate([1,2,3]):
        dfsub = df.loc[df['class']==str(class_idx)]
        plt.errorbar(np.arange(10)+i*0.1, -dfsub['dprob mean'], yerr=dfsub['sigma'], fmt='o', zorder=10, color=COLORS[i])
    
    plt.legend(['Healthy', 'Apoptosis', 'Dead'])
    plt.xticks(np.arange(10), dfsub['channel'], rotation=-65)
    plt.xlabel("(b) Split by true class")
    plt.ylabel("Percentage drop")
    plt.tight_layout()
    plt.savefig("channel_occlusion_drops_avg.pdf")
    plt.savefig("channel_occlusion_drops_avg.png", dpi=250)
    
    plt.show()
    