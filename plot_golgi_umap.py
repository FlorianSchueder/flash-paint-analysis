import os

import anndata as ad
import scanpy as sc

import umap

import matplotlib.pyplot as plt
from matplotlib import colormaps

from util import log, find_paths, preprocess, SPLIT_KEY

channels_directory = r"D:\FLASH-PAINT\zzzz.for_Zach\GolgiPlex\z.Drugs"
debug = True
overwrite =  True
downsample = None
n_neighbors = 50
min_dist = 0.1
pool_by = None
# pool_by = ["Normal", "BrefeldinA", "Ilimaquinone", "Nocodazole"]
n_subsamp = 1000000
min_locs = 30
stratified = False
plot_targets = ["GM130","GRASP65","GRASP55","ManII","p230","Golgin97","Giantin","TGN46","COPI","ERGIC53","COPII","Tango1","Lamin"]

def plot_umap(df, color=None, save=None, cmap=None, title=None, size=None , ax=None):
    if color is None:
        color = df.obs['target'].unique()
    elif isinstance(color, str):
        color = df.obs[color]
    
    if cmap is None:
        try:
            from scanpy.plotting.palettes import vega_20
            cmap = vega_20
        except ImportError:
            cmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    else:
        try:
            cmap = colormaps[cmap].resampled(len(color)).colors
        except AttributeError:
            cmap = [colormaps[cmap].resampled(len(color))(x) for x in range(len(color))]
    
    was_none = False
    if ax is None:
        was_none = True
        fig, ax = plt.subplots()

    cmap = cmap[:len(color)][::-1]
    
    if size is None:
        size = max(120000/df.shape[0], 0.001)
    for i, target in enumerate(color):
        x, y = df.obsm['X_umap'][df.obs['target'] == target].T
        ax.scatter(x, y, marker='.', c=cmap[i], s=size, rasterized=False, plotnonfinite=True)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)

    for i, target in enumerate(color):
        ax.scatter([], [], c=cmap[i] ,label=target)
    ax.legend(loc='center left', frameon=False, bbox_to_anchor=(1, 0.5), fontsize=None)

    ax.autoscale_view()
    
    if was_none:
        fig.tight_layout()
    
    if isinstance(save, str):
        plt.savefig(save, dpi=300)
    elif save is not None:
        raise ValueError(f"Unknown what to do. Expected file path, but recieved {save}.")

if __name__ == "__main__":
    channels_directories = find_paths(channels_directory, exclude="ROI")

    if pool_by is not None:
        pools = [[d for d in channels_directories if x in d and "Tango" not in d] for x in pool_by]
    else:
        pools = [[d] for d in channels_directories]
        pool_by = ['_'.join(d.split(SPLIT_KEY)[-2:]) for d in channels_directories]

    for pool, data_name in zip(pools, pool_by):
        log(pool, debug=debug)
        adatas = {}
        for channels_directory in pool:
            name = '_'.join(channels_directory.split(SPLIT_KEY)[-2:])

            log(f"Processing {name}", debug=debug)
            
            if not os.path.exists(f"features_{name}.h5ad"):
                log(f"could not find file features_{name}.h5ad")
            
            # Read in data
            df = ad.read(f"features_{name}.h5ad")
            log(f"Total points: {df.X.shape[0]}", debug=debug)
            
            if overwrite or df.obsm.get("X_umap") is None:
                # Drop regions with particularly low counts
                df.obs['n_locs'] = df.X.sum(1)
                adatas[channels_directory] = df[df.obs['n_locs'] > min_locs]

                log(f"After filtering: {adatas[channels_directory].X.shape[0]}", debug=debug)

                # Subsample
                if stratified:
                    split_n = n_subsamp // (len(pool)*df.X.shape[1]) #/ adatas[channels_directory].X.shape[0]
                    inds = adatas[channels_directory].obs.groupby('target').apply(lambda x: x.sample(min(split_n, len(x)))).droplevel(0).index
                    adatas[channels_directory] = adatas[channels_directory][inds]
                else:
                    sc.pp.subsample(adatas[channels_directory], n_obs=n_subsamp, copy=False)

                # deal with variability in naming 
                adatas[channels_directory].obs['target'].replace({"Grasp65": "GRASP65", "Grasp55": "GRASP55", "ManII-GFP": "ManII"}, inplace=True)

                log(f"After subsample: {adatas[channels_directory].X.shape[0]}", debug=debug)

                adatas[channels_directory].X = preprocess(adatas[channels_directory].X, target_sum=10000, downsample=downsample)
                    
            else:
                adatas[channels_directory] = df

        data = ad.concat(adatas, join="inner")

        print([x for x in data.obs['target'].unique()])

        write_again = False
        if overwrite or data.obsm.get("X_umap") is None:
            try:
                embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, low_memory=False, init='pca', verbose=True).fit_transform(data.X)
                data.obsm["X_umap"] = embedding
                write_again = True
                log(f"SUCCESS! {name} umap", debug=debug)
            except Exception as e:
                log(f"{name} umap FAILED: {e}", debug=debug)
                write_again = False
        else:
            embedding = data.obsm["X_umap"]

        try:
            sc.pl.umap(data, title=data_name, color="target", show=False, size=max(120000/embedding.shape[0], 0.01), save=f"features_{data_name}_umap.pdf")
            plot_umap(data, color=plot_targets, save=f"features_{data_name}_umap.pdf", title=data_name)
        except KeyError as e:
            log(f"Could not plot: {e}", debug=debug)

        if write_again:
            data.write(f"features_{data_name}_umap.h5ad", compression="gzip")
    
