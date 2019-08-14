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
    dataset = {'X': np.array(X1), 'y': np.array(y1) }
    if _dump:
        with open(os.path.join(datadir, 'combined_dataset.pkl'), 'wb') as pkl:
            dump(dataset, pkl)
    return dataset

if __name__ == "__main__":
    combine_dataset(_dump=True)
