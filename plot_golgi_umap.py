import os

import anndata as ad
import scanpy as sc

import umap

from util import log, find_paths, preprocess, SPLIT_KEY

channels_directory = r"D:\FLASH-PAINT\zzzz.for_Zach\GolgiPlex\z.Drugs"
# channels_directory = r"D:\FLASH-PAINT\zzzz.for_Zach\GolgiPlex\z.Drugs\Normal\231028_Normal"
debug = True
overwrite =  True
downsample = None
n_neighbors = 50
min_dist = 0.1
pool_by = None
pool_by = ["Normal", "BrefeldinA", "Ilimaquinone", "Nocodazole"]
n_subsamp = 2000000
min_locs = 30
stratified = True

if __name__ == "__main__":
    channels_directories = find_paths(channels_directory, exclude="ROI")

    if pool_by is not None:
        pools = [[d for d in channels_directories if x in d and "Tango" not in d] for x in pool_by]
    else:
        pools = [[d] for d in channels_directories]
        pool_by = ['_'.join(d.split(SPLIT_KEY)[-2:]) for d in channels_directories]

    for pool, data_name in zip(pools, pool_by):
        adatas = {}
        for channels_directory in pool:
            name = '_'.join(channels_directory.split(SPLIT_KEY)[-2:])

            log(f"Processing {name}", debug=debug)
            
            if not os.path.exists(f"features_{name}.h5ad"):
                log(f"could not find file features_{name}.h5ad")
            
            # Read in data
            df = ad.read(f"features_{name}.h5ad")
            log(f"Total points: {df.X.shape[0]}", debug=debug)
            
            # cleanup hdf5 name (bug)
            # df.obs['target'] = df.obs['target'].cat.rename_categories(lambda x: x.split('.hdf5')[0])
            
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

                log(f"After subsample: {adatas[channels_directory].X.shape[0]}", debug=debug)

                # deal with variability in naming
                adatas[channels_directory].var.replace({"Grasp65": "GRASP65", "ManII-GFP": "ManII"}, inplace=True)

                adatas[channels_directory].X = preprocess(adatas[channels_directory].X, target_sum=10000, downsample=downsample)
                    
            else:
                adatas[channels_directory] = df

        data = ad.concat(adatas, join="inner")

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
        except KeyError as e:
            log(f"Could not plot: {e}", debug=debug)

        if write_again:
            data.write(f"features_{data_name}_umap.h5ad", compression="gzip")
    
