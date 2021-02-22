import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wfutils import CHANNELS


RANKINGS = [
    r"D:\Speciale\Code\output\SimpleML\Ranking Entropy - Real (Correct)\ranking.csv", 
    r"D:\Speciale\Code\output\SimpleML\Ranking Correlation - Real (Correct)\ranking.csv", 
    r"D:\Speciale\Code\output\ChannelOcclusion\Real\Run1\ranking.csv", 
    '4807265391']

HEADERS = [
    "Entropy", 
    "Correlation", 
    "Channel Occlusion", 
    "Teacher-Student"]


if __name__ == "__main__":
    # Load rankings
    rankings = []
    for i in range(len(RANKINGS)):
        if os.path.isfile(RANKINGS[i]):
            fpath = RANKINGS[i]
            ranking = pd.read_csv(fpath, header=None)
            if ranking.values.shape[0] > 10:
                ranking = pd.read_csv(fpath, header='infer')
            if(ranking.values.shape[1]==1):
                ranking = np.array(ranking.values[:,0], dtype=int)
            elif(ranking.values.shape[1]==2):
                ranking = ranking.values[:,0]
            elif(ranking.values.shape[1]==4):
                ranking = ranking.values[:,-1]
            else:
                raise Exception("Ranking file has invalid number of columns.")
        elif len(RANKINGS[i])==10:
            ranking = np.array([int(c) for c in RANKINGS[i]])
        else:
            raise Exception("Invalid ranking file/order specification")
        
        rankings.append(ranking)

    # Create corresponding channel names
    ch_names = []
    for ranking in rankings:
        names = [CHANNELS[idx] for idx in ranking]
        ch_names.append(names)
    
    #print(rankings)
    #print(ch_names)
    f = open("latex_table_rankings.txt", 'w')
    f.write(r"\begin{table}[t]" + "\n")
    f.write(r"    \centering" + "\n")
    f.write(r"    \begin{tabular}{c " + " ".join(["l" for _ in rankings]) + "}\n")
    f.write(r"    \hline" + "\n")
    f.write("     & " + " & ".join([header for header in HEADERS]) + r"\\" + "\n")
    f.write(r"    \hline" + "\n")
    for line_idx in range(10):
        f.write("    %d & " % (line_idx+1))
        line_elements = []
        for ele in range(len(rankings)):
            ch_num = rankings[ele][line_idx]
            ch_name = ch_names[ele][line_idx]
            line_elements.append("%d (%s)" % (ch_num, ch_name))
        f.write(" & ".join(line_elements))    
        f.write(r"\\" + "\n")

    f.write(r"    \hline" + "\n")
    f.write(r"    \end{tabular}" + "\n")
    f.write(r"    \caption{Channel ranking using the information entropy measure and the two variants of the correlation measure.}" + "\n")
    f.write(r"    \label{tab:INSERT_CORRECT_LABEL}" + "\n")
    f.write(r"\end{table}")
    f.close()


    f = open("latex_table_subsets.txt", 'w')
    f.write(r"\begin{table}[t]" + "\n")
    f.write(r"    \centering" + "\n")
    f.write(r"    \begin{tabular}{c " + " ".join(["l" for _ in rankings]) + "}\n")
    f.write(r"    \hline" + "\n")
    f.write("     & " + " & ".join([header for header in HEADERS]) + r"\\" + "\n")
    f.write(r"    \hline" + "\n")
    for line_idx in range(10):
        f.write("    %d & " % (line_idx+1))
        line_elements = []
        for ele in range(len(rankings)):
            ch_nums = ",".join(str(n) for n in sorted([num for num in rankings[ele][:(line_idx+1)]]))
            line_elements.append("%s" % ch_nums)
        f.write(" & ".join(line_elements))    
        f.write(r"\\" + "\n")

    f.write(r"    \hline" + "\n")
    f.write(r"    \end{tabular}" + "\n")
    f.write(r"    \caption{Channel ranking using the information entropy measure and the two variants of the correlation measure.}" + "\n")
    f.write(r"    \label{tab:INSERT_CORRECT_LABEL}" + "\n")
    f.write(r"\end{table}")
    f.close()