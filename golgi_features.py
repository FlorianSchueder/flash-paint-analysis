import os
import glob
import time

import numpy as np
import pandas as pd
import anndata as ad

import dask
from dask.delayed import delayed
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client

from sklearn.neighbors import KDTree

from util import log, find_paths, SPLIT_KEY

## dask configuration setup
## bump timeouts to keep workers alive
dask.config.set({"distributed.deploy.lost-worker-timeout": "30s"})  # default, 15s
dask.config.set({"distributed.comm.timeouts.connect": "60s"})  # default, 30s
dask.config.set({"distributed.comm.timeouts.tcp": "60s"})  # default, 30s

### User parameters

# channels_directory = r"H:\09.Adapter_PAINT\zzzz.for_Zach\GolgiPlex\z.Drugs"
# channels_directory = "/Users/zachmarin/Documents/Projects/FlashPAINT/GolgiPlex/Normal_Golgi"
# channels_directory = r"D:\FLASH-PAINT\zzzz.for_Zach\GolgiPlex\z.Drugs"
channels_directory = r"D:\FLASH-PAINT\zzzz.for_Zach\GolgiPlex\z.Drugs\Normal\231028_Normal"

clipping = None  # (35869, 10472, -379, 44766, 19660, 112)  # xl, yl, zl, xu, yu, zu
max_rad = 100  # maximum radius from each point
chunksize = 5000000
overwrite = True  # overwrite feature files
debug = True
pixel_size = 108

# End user parameters ###

pixels = np.array([pixel_size, pixel_size, 1])
pixels = da.from_array(pixels)

channels_directories = find_paths(channels_directory, exclude="ROI")
print(channels_directories)

# def query_ball_point(points, tree, rad):
#     _, dist = tree.query_radius(points, rad, count_only=False, return_distance=True)
#     return da.array([da.mean(x) if len(x) > 0 else rad for x in dist], dtype=float)

def query_ball_point(points, tree, rad):
    return da.asarray([x for x in tree.query_radius(points, rad, count_only=True)], dtype=int)

if __name__ == '__main__':

    # n_cpus = os.cpu_count() - 1
    client = Client(processes=False) #n_workers=n_cpus, threads_per_worker=1, memory_limit="8GiB")
    client.get_versions(check=True)
    print(client.dashboard_link)

    for channels_directory in channels_directories:
        name = '_'.join(channels_directory.split(SPLIT_KEY)[-2:])

        log(f"Processing {name}", debug=debug)
        start = time.time()

        # Get the list of channel files in this directory
        channel_files = glob.glob(f"{channels_directory}/*hdf5")
        true_names = []
        points = {}
        n_points = {}
        total_points = 0

        # Open the files
        for i, ch_fp in enumerate(channel_files):
            if "all" in ch_fp.lower():
                # Don't load the combined channels
                continue

            key = os.path.basename(ch_fp).split('_')[-1].split('.hdf5')[0]
            log(f"{key}", debug=debug)
            true_names.append(key)
            
            points[key] = dd.read_hdf(ch_fp, key="locs", chunksize=chunksize)[["x", "y", "z"]].to_dask_array(lengths=True)*pixels[None,:]
            
            # Create a KDTree for each channel
            n_points[key] = points[key].shape[0]
            total_points += n_points[key]

        # Guarantee consistent ordering of the names
        sorted_ixs = np.argsort(true_names)
        true_names = np.array(true_names)[sorted_ixs]

        channel_trees = []
        for i, ch in enumerate(true_names):
            channel_trees.append(delayed(KDTree)(points[ch]))

        channel_trees = client.persist(channel_trees)
            
        log(f"Total points: {total_points}", debug=debug)

        index = 0
        # Construct an n_points x n_channels*len(rads) feature matrix
        features = da.zeros((total_points, len(true_names)), dtype=int)
        # feature_channel = np.zeros((total_points,), dtype=int)

        for j, ch in enumerate(true_names):
            for i, ch2 in enumerate(true_names):
                features[index:(index+n_points[ch]), i] = points[ch].map_blocks(query_ball_point, channel_trees[i], max_rad, drop_axis=1, dtype=int) 
            # feature_channel[index:(index+n_points[ch])] = j
            index += n_points[ch] # shift to the next part of the array


        df = ad.AnnData(features.compute(), dtype=features.dtype)
        df.var_names = [f"{var}" for var in true_names]  # feature names
        # df.obs['target'] = pd.Categorical(true_names[feature_channel]) # target names
        df.obs['target'] = pd.Categorical([name for name in true_names for k in range(points[name].shape[0])])
        df.obsm['spatial'] = da.vstack([points[k] for k in true_names]).compute()
        df.uns['filenames'] = [ch_fp for ch_fp in channel_files if not "all" in ch_fp.lower()]  # store source file names

        df.write_h5ad(f"features_{name}.h5ad") #, compression="gzip")

        stop = time.time()
        duration = stop - start
        log(f"{name} took {duration:.2f} s", debug=debug)
        start = time.time()
            
    client.close()

  