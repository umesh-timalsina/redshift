from pickle import load, dump
import numpy as np
import glob
import os


def combine_dataset(part='dataset-PART*.pkl', 
                    datadir='../data/images',
                    _dump=False):
    """Combine the multipart data into one
        and dump to a single file"""
    path_glob = os.path.join(datadir, part)
    print(path_glob)
    data_parts = sorted(glob.glob(path_glob))
    X1 = []
    y1 = []
    for i in range(len(data_parts)):
        with open(data_parts[i], 'rb') as _part_pkl:
            data_part = load(_part_pkl)
            X1.extend(data_part['X'])
            y1.extend(data_part['y'])
    print(np.array(X1).shape)

    if _dump:
        with open(os.path.join(data_dir, 'combined_dataset.pkl'), 'wb') as pkl:
            dump(X1, pkl)
    return {
        'X': X1,
        'y': y1
    }

if __name__ == "__main__":
    combine_dataset()
