import os
import sys

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import anndata as ad
import scanpy as sc

import umap

from util import log, find_paths, preprocess, SPLIT_KEY

channels_directory = r"D:\FLASH-PAINT\zzzz.for_Zach\GolgiPlex\z.Drugs"
debug = True
overwrite =  True
downsample = None
n_neighbors = 50
min_dist = 0.1

if __name__ == "__main__":
    channels_directories = find_paths(channels_directory, exclude="ROI")

    _components = None
    for channels_directory in channels_directories:
        name = '_'.join(channels_directory.split(SPLIT_KEY)[-2:])

        log(f"Processing {name}", debug=debug)
        
        if not os.path.exists(f"features_{name}.h5ad"):
            log(f"could not find file features_{name}.h5ad")
        
        
        # Read in data
        df = ad.read(f"features_{name}.h5ad")
        log(f"Total points: {df.X.shape[0]}", debug=debug)
        
        # cleanup hdf5 name (bug)
        df.obs['target'] = df.obs['target'].cat.rename_categories(lambda x: x.split('.hdf5')[0])
        
        write_again = False
        if overwrite or df.obsm.get("X_umap") is None:
            # Drop regions with particularly low counts
            df.obs['n_locs'] = df.X.sum(1)
            median_counts_per_obs = df.obs['n_locs'].median()
            #mad_counts_per_obs = np.median(np.abs(df.obs['n_locs'] - median_counts_per_obs))
            min_counts = 30 # (median_counts_per_obs - mad_counts_per_obs)
            df2 = df[df.obs['n_locs'] > min_counts]

            print(f"After filtering: {df2.X.shape[0]}")

            # Subsample
            sc.pp.subsample(df2, n_obs=1000000, copy=False)

            print(f"After subsample: {df2.X.shape[0]}")

            X_norm = preprocess(df2.X, downsample=downsample)
            
            try:
                embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, low_memory=False, init='pca', verbose=True).fit_transform(X_norm)
                df2.obsm["X_umap"] = embedding
                write_again = True
                log(f"SUCCESS! {name} umap", debug=debug)
            except Exception as e:
                log(f"{name} umap FAILED: {e}", debug=debug)
                write_again = False
                
        else:
            df2 = df
            embedding = df2.obsm["X_umap"]
        
        try:
            sc.pl.umap(df2, title=name, color="target", show=False, size=max(120000/embedding.shape[0], 0.01), color_map="viridis", save=f"features_{name}_umap.pdf")
        except KeyError as e:
            log(f"Could not plot: {e}", debug=debug)

        if write_again:
            df2.write(f"features_{name}_umap.h5ad", compression="gzip")
        
