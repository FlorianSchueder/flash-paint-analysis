# FLASH-PAINT Analysis
Repository for analysis performed in [Unraveling cellular complexity with unlimited multiplexed superresolution imaging](https://www.biorxiv.org/content/10.1101/2023.05.17.541061v1).

## Setup

```
conda create -n golgi python=3.9
pip install numpy anndata scanpy umap-learn scikit-learn dask[complete] matplotlib pandas tables
```

## Files

***golgi_features.py*** - Computes feature matrices from FLASH-PAINT images in Picasso format.

***plot_golgi_matrices.py*** - Creates clustermap from ROIs of FLASH-PAINT images in Picasso format.

***plot_golgi_umap.py*** - Computes and plots UMAP embedding of features generated by golgi_features.py.

***plot_umap_and_image.py*** - Plot UMAP next to original image with ROI highlighted.

**cluster_analysis.py** - Tools for clustering and investigating UMAP embedding.

***util.py*** - Common functions used by all scripts.
