import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import HDBSCAN

def cluster_umap(df, min_cluster_size=20, copy=True):
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=-1)
    hdb.fit(df.obsm["X_umap"])

    if copy:
        df2 = df.copy()
        df2.obs["labels"] = hdb.labels_
        return df2
    else:
        df.obs["labels"] = hdb.labels_
        return df

def make_knee_plot(df, sizes = [5, 10, 20, 40, 80, 160, 320], return_clustering=False):
    n_clusters = []
    if return_clustering:
        dfs = []
        copy = True
    else:
        copy = False
    for c in sizes:
        df = cluster_umap(df, min_cluster_size=c, copy=copy)
        n_clusters.append(len(np.unique(df.obs["labels"])))
        if return_clustering:
            dfs.append(df)

    plt.plot(sizes, n_clusters)
    plt.xlabel('Minimum cluster size')
    plt.ylabel('Number of clusters')
    plt.savefig("knee_plot.png")

    if return_clustering:
        return dfs

def get_ranked_targets(df, n=5, min_count=30, min_targets=2, include=None, exclude=None):
    # How many points are in each cluster?
    labels, counts = np.unique(df.obs["labels"], return_counts=True)

    # Sort the labels from largest to smallest cluster
    labels_sorted = labels[np.argsort(counts)[::-1]]
    labels_sorted = labels_sorted[labels_sorted>=0]
    
    if not isinstance(include, list):
        include = [include]
    if not isinstance(exclude, list):
        exclude = [exclude]
    
    target_clusters = []
    
    stop_at = n
    for label in labels_sorted:
        if stop_at == 0:
            break
        
        targets = sorted([k for k, v in df.obs['target'][df.obs["labels"] == label].value_counts().to_dict().items() if v > min_count])

        if len(targets) < min_targets:
            continue
        
        found_included = False
        for target in targets:
            if target in include:
                found_included = True
                break

        found_excluded = False
        for target in targets:
            if target in exclude:
                found_excluded = True
                break

        if found_included and not found_excluded and targets not in target_clusters:
            target_clusters.append(targets)
            stop_at -= 1
            
    return target_clusters

if __name__ == "__main__":
    import anndata as ad

    fn = "features_BrefeldinA_umap1.h5ad"
    df = ad.read(fn)

    make_knee_plot(df, sizes = [20, 40, 80, 160, 320, 640, 1280] )