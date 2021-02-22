import os

def create_file(filepath, content):
    f = open(filepath, 'w')
    f.write(content)
    f.close()


def get_sh_text(chs):
    text = "#!/bin/sh\n#BSUB -q gpuv100\n#BSUB -J C"
    text += "".join([str(ch) for ch in sorted(chs)]) + "\n"
    text += "#BSUB -n 8\n#BSUB -gpu \"num=1:mode=exclusive_process\"\n#BSUB -W 23:59\n#BSUB -R \"rusage[mem=3GB]\"\n#BSUB -u tore.hilbert@hotmail.com\n#BSUB -B\n#BSUB -N\n#BSUB -o gpu-%J.out\n#BSUB -e gpu_%J.err\n"
    text += "\n"
    text += "module load python3/3.7.5\nmodule load cuda/10.0\nmodule load cudnn/v7.6.5.32-prod-cuda-10.0\n"
    text += "\n"
    text += "python3 ~/Speciale/Code/sessions/train_manual.py -path_training_data HPC"
    text += " -epochs=120 -lr_steps 50 70 90 -aug_noise=0.25 -weight_decay_l2=0.0001"
    text += " -resnet_n_channels=%d" % len(chs)
    text += " -use_channels " + " ".join([str(ch) for ch in sorted(chs)])
    text += " -resnet_block_sizes 3 5 3 -resnet_filter_size 32 64 128"
    return text


def get_sh_filename(chs):
    chs = sorted(chs)
    return "train_" + "_".join([str(ch) for ch in chs]) + ".sh"


def if_two_of_same(chs):
    already = []
    for ch in chs:
        if ch in already:
            return True
        already.append(ch)
    return False



if __name__ == "__main__":
    channels = [0,1,2,3,4,5,6,7,8,9]
    pdir = os.path.dirname(os.path.abspath(__file__))



    # 4 CHANNEL COMBS  
    combs = {}
    for ch1 in range(10):
        for ch2 in range(10):
            for ch3 in range(10):
                for ch4 in range(10):
                    if if_two_of_same([ch1,ch2,ch3,ch4]):
                        continue

                    chs = sorted([ch1, ch2, ch3, ch4])
                    chsstr = " ".join([str(ch) for ch in chs])
                    if chsstr not in combs:
                        combs[chsstr] = ""
                        filepath = os.path.join(pdir, get_sh_filename(chs))
                        content = get_sh_text(chs)
                        create_file(filepath, content)

    # 3 CHANNEL COMBS
    # combs = {}
    # n = 0
    # for ch1 in range(10):
    #     for ch2 in range(10):
    #         for ch3 in range(10):
    #             if ch1==ch2 or ch1==ch3 or ch2==ch3:
    #                 continue
    #             chs = sorted([ch1, ch2, ch3])
    #             chsstr = " ".join([str(ch) for ch in chs])
    #             if chsstr not in combs:
    #                 combs[chsstr] = ""
    #                 filepath = os.path.join(pdir, get_sh_filename(chs))
    #                 content = get_sh_text(chs)
    #                 create_file(filepath, content)
    
    # 2 CHANNEL COMBS
    # for ch1 in range(10):
    #     for ch2 in range(ch1+1, 10):
    #         chs = [ch1,ch2]
    #         filepath = os.path.join(pdir, get_sh_filename(chs))
    #         content = get_sh_text(chs)
    #         create_file(filepath, content)

                
