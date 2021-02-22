import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wfutils.mmap
from wfutils.progess_printer import ProgressPrinter


if __name__ == "__main__":
    mmaps = wfutils.mmap.get_class_mmaps_read(r"E:\validate32_redist")

    printer = ProgressPrinter(
        steps=mmaps[0].shape[0] + mmaps[1].shape[0] + mmaps[2].shape[0], 
        header='Progress', sign_start=" |", sign_end="|", sign_tick="-", print_evolution_number=False)
    printer.start()

    zero_images = {}
    for i,mmap in enumerate(mmaps):
        for j in range(mmap.shape[0]):
            printer.step()
            vmin = np.min(mmap[j])
            vmax = np.max(mmap[j])
            if (vmax - vmin) < 0.05:
                zero_images['%d %d' % (i, j)] = (vmin, vmax)
    print("")
    print(zero_images)