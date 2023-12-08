import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scanpy.plotting.palettes import vega_20
import numpy as np

import os
import glob

import anndata as ad
from plot_golgi_umap import plot_umap

def get_original_points(df, u0l, u0u, u1l, u1u):
    x, y = df.obsm['X_umap'].T
    inds = (x > u0l) & (x < u0u) & (y > u1l) & (y < u1u)
    xs, ys, zs = df.obsm['spatial'][inds].T
    targets = df.obs["target"][inds]
    
    return xs, ys, zs, targets

if __name__ == "__main__":
    plt.style.use('dark_background')

    fn = "features_Normal_231116_Normal_umap.h5ad"
    df = ad.read(fn)

    plot_targets = ["GM130","GRASP65","GRASP55","ManII","p230","Golgin97","Giantin","TGN46","COPI","ERGIC53","COPII","Tango1","Lamin"]


    cmap = vega_20
    cmap = cmap[:len(plot_targets)][::-1]

    a, b, c = df.obsm['spatial'].T
    ranges = [(30, 31, 3, 4),
            (27, 28, 13.5, 15),
            (24, 26, -11, -9),
            (9.3, 11.2, 3.5, 5.3),
            (7.5, 9, 2, 3.2),
            (4.65, 6.6, 1.3, 3),
            (2, 4, -1.2, 0.2),
            (0.75, 1.75, -3.75, -2.5),
            (-1.5, 0.25, -6.5, -4),
            (-6, -2.8, -10.5, -7.5),
            (-0.75, 0.5, -10.5, -7.8),
            (15, 19, -1, 5),
            ]


    for xl, xu, yl, yu in ranges:
        fig, axs = plt.subplots(1,2, figsize=(10,5))
        plot_umap(df, color=plot_targets, size=0.0001, ax=axs[0])
        rect = Rectangle((xl, yl), xu-xl, yu-yl, linewidth=3, edgecolor='white', facecolor='none')
        axs[0].add_patch(rect)
        axs[1].scatter(a, b, c=c, cmap='binary_r', s=0.0001)
        
        xs, ys, zs, targets = get_original_points(df, xl, xu, yl, yu)
        tt = []
        for target in targets.unique():
            i = plot_targets.index(target)
            new_idxs = targets == target
            if np.sum(new_idxs) < 0.1*len(targets):
                continue
            tt.append(target)
            axs[1].scatter(xs[new_idxs], ys[new_idxs], c=cmap[i], s= 0.01)
        if len(tt) > 5:
            tt.insert(5,'\n')
        axs[1].set_title(" ".join(tt))
        axs[1].axis('off')
        axs[1].autoscale_view()
        fig.tight_layout()
        
        plt.savefig(f'umap_{xl}_{xu}_{yl}_{yu}.pdf')
