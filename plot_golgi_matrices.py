import os
import glob
import json

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from util import log, find_paths, preprocess, SPLIT_KEY

### User parameters
channels_directory = r"D:\FLASH-PAINT\zzzz.for_Zach\GolgiPlex\z.Drugs"
concentration_fp = "concentrations.json"
debug = True
overwrite = True
false_count = 999999
pool_by = ["BrefeldinA", "Ilimaquinone", "Nocodazole"]
plot_targets = ["GM130","GRASP65","GRASP55","ManII","p230","Golgin97","Giantin","TGN46","COPI","ERGIC53","COPII","Tango1"] #,"Lamin"]
# End user parameters ###

if __name__ == "__main__":
    channels_directories = find_paths(channels_directory, include="ROI")
    
    with open(concentration_fp, "r") as fp:
        c = json.load(fp)

    if pool_by is not None:
        pools = [[d for d in channels_directories if x in d] for x in pool_by]
    else:
        pools = channels_directories
        pool_by = channels_directories

    for pool, data_name in zip(pools, pool_by):
        adatas = {}
        last_roi = 0
        for channels_directory in pool:
            name = '_'.join(channels_directory.split(SPLIT_KEY)[-3:-1])
            if "Tango" in name:
                continue
            log(name, debug=debug)
            
            pickprops = glob.glob(os.path.join(channels_directory, "*apicked_pyme2_d_nucleus.hdf5"))
                    
            # Load in data
            dfs = {}
            n_target = 0
            n_obs = 0

            for pp in pickprops:
                target = os.path.splitext(os.path.basename(pp))[0].split("_")[2]
                if target == "ALL":
                    continue
                if target == "Lamin":
                    continue
              
                dfs[target] = pd.read_hdf(pp, key='locs')
                n_target += 1
                group = dfs[target]['group']
                n_obs = max(n_obs, np.max((dfs[target]['group'].value_counts() != false_count).index))
                
            # Construct a matrix of counts per ROI for each target
            counts = np.zeros((n_obs+1, n_target), dtype=int)
            dist = np.zeros((n_obs+1, n_target))
            plot_vars = []
            for i, target in enumerate(dfs.keys()):
                group = dfs[target]['group']
                n_events = group.value_counts(sort=False)
                rows = n_events != false_count
                counts[rows.index,i] = n_events[rows.index].values
                for j in rows.index:
                    rows2 = dfs[target]['group'] == j
                    dist[j, i] = np.median(dfs[target]['d_nucleus'][rows2])
                plot_vars.append(target)
                
            # Construct AnnData structure of counts
            adatas[channels_directory] = ad.AnnData(counts)
            adatas[channels_directory].var_names = plot_vars
            adatas[channels_directory].obs['roi'] = pd.Categorical([str(x) for x in range(last_roi, last_roi+n_obs+1)])
            adatas[channels_directory].obs['dist'] = dist.mean(1)
            last_roi += n_obs + 1

            # scale by concentration
            try:
                conc = c[[x for x in c.keys() if x in channels_directory][0]]
            except IndexError:
                log(f"Nothing found in {concentration_fp} for {name}", debug=debug)
                continue
            conc_max = max(conc.values())
            for k, v in conc.items():
                try:
                    adatas[channels_directory][:,k].X = (adatas[channels_directory][:,k].X*v/conc_max).ravel()
                except KeyError as e:
                    if k == "GRASP65":
                        k = "Grasp65"
                        adatas[channels_directory][:,k].X = (adatas[channels_directory][:,k].X*v/conc_max).ravel()
                    elif k == "Grasp65":
                        k = "GRASP65"
                        adatas[channels_directory][:,k].X = (adatas[channels_directory][:,k].X*v/conc_max).ravel()
                    elif k == "Lamin":
                        continue
                    else:
                        raise e
           
            # deal with variability in naming
            adatas[channels_directory].var.rename({"Grasp65": "GRASP65", "ManII-GFP": "ManII"}, inplace=True)
            
            adatas[channels_directory].X = preprocess(adatas[channels_directory].X, target_sum=None)
            # adatas[channels_directory].X = PowerTransformer().fit_transform(adatas[channels_directory].X)
            # adatas[channels_directory].X = (adatas[channels_directory].X - adatas[channels_directory].X.mean(0)[None,:])/adatas[channels_directory].X.std(0)[None,:]
            
        
        data = ad.concat(adatas, join="inner")
        data.obs_names = data.obs['roi']

        # plot
        ar = data.X.shape[0]/len(plot_targets)
        #mp = sc.pl.matrixplot(data, plot_targets, groupby='roi', dendrogram=True, cmap='RdBu_r', categories_order=data.obs['roi'][data.obs['dist'].sort_values().index], swap_axes=True, show=False, save=f'{data_name}_matrixplot.png', vmin=-2, vmax=2, colorbar_title="Z-Score") #, standard_scale="var")
        mp = sc.pl.clustermap(data.T, cmap='RdBu_r', show=False, save=f'{data_name}_clustermap.png', figsize=(10*ar,10), vmin=-2, vmax=2)
        #sc.pl.heatmap(data, plot_targets, groupby='roi', dendrogram=True, cmap='RdBu_r', swap_axes=True, show=False, save=f'{data_name}_matrixplot.png', vmin=-2, vmax=2)#, colorbar_title="Z-Score") #, standard_scale="var")
 