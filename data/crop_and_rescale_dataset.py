import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.mmap
import wfutils.infofile
from wfutils.progess_printer import ProgressPrinter


PATH_TRAIN_DATA = r"E:\validate32_redist"

PATH_TRAIN_DATA_OUT = r"E:\validate32_redist_crop_rescale"


if __name__ == "__main__":

    if not os.path.isdir(PATH_TRAIN_DATA_OUT):
        os.makedirs(PATH_TRAIN_DATA_OUT)
    
    mmaps = wfutils.mmap.get_class_mmaps_read(PATH_TRAIN_DATA)

    # A) Channel-wise percentile values (excluding background)
    print("Calculating percentiles for transformation:")
    force_calculate_percentiles = False
    path_transformation = os.path.join(PATH_TRAIN_DATA_OUT, 'transformation.npy')

    if not os.path.isfile(path_transformation) or force_calculate_percentiles:
        pall = np.zeros(shape=(3,2,10), dtype="float32")

        printer = ProgressPrinter(10, header="Class")
        for i,mmap in enumerate(mmaps):
            printer.start()
            ps = np.zeros(shape=(2,10), dtype="float32")
            for ch in range(10):
                X = mmap[:,16:48,16:48,ch]
                X_masked = np.ma.masked_where(X==0, X)
                X_masked = np.ma.filled(X_masked, np.nan)
                ps[:,ch] = np.nanpercentile(X_masked, [1,99])
                
                printer.step()
            pall[i,:,:] = ps

        pall = np.mean(pall, axis=0, dtype="float32")   # (3,2,10) -> (2,10)
        np.save(path_transformation, pall)
    else:
        pall = np.load(path_transformation)

    # B) Transformation and crop images (y=ax+b)
    print("Constructing new dataset:")
    file_info = open(os.path.join(PATH_TRAIN_DATA_OUT, 'info.txt'), 'w')
    vmin_desired = 0.025
    vmax_desired = 1.00
    for i,mmap in enumerate(mmaps):
        N = mmap.shape[0]
        batch_size = 256
        batches = (N // batch_size) + 1
        cursor = 0

        printer = ProgressPrinter(steps=batches, header="Class %d" % (i+1), print_evolution_number=False)
        printer.start()
        file_info.write("%d,%d,%d,%d\n" % (N, 32, 32, 10))
        mmap_out = np.memmap(os.path.join(PATH_TRAIN_DATA_OUT, "%d.npy" % (i+1)), dtype="float32", mode='w+', shape=(N, 32, 32, 10))
        for batch in range(batches):
            # set cursor end
            cursor_end = cursor + batch_size
            if(cursor_end > N):
                cursor_end = N
            
            # load, crop, and transform data
            X = mmap[cursor:cursor_end, 16:48, 16:48, :]
            X_masked = np.ma.masked_where(X==0, X)
            for ch in range(10):
                X_masked[:,:,:,ch] = (vmax_desired - vmin_desired) * ((X_masked[:,:,:,ch] - pall[0,ch])/(pall[1,ch] - pall[0,ch])) + vmin_desired
            X_masked = np.ma.filled(X_masked, 0)

            # insert into new mmap
            mmap_out[cursor:cursor_end,:,:,:] = X_masked

            # increment cursor
            cursor = cursor_end
            printer.step()
    
    file_info.write("float32")
