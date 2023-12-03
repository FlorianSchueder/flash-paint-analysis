import os
import sys

SPLIT_KEY = '\\' if sys.platform == "win32" else '/'

def log(msg, debug=False):
    if debug:
        print(msg)

def find_paths(dirname, exclude=[], include=[]):
    if isinstance(exclude, str):
        exclude = [exclude]
    if isinstance(include, str):
        include = [include]
    paths = []
    for dirpath, _, filenames in os.walk(dirname):
        for file in filenames:
            skip = False
            for ex in exclude:
                if ex in dirpath:
                    skip = True
                    break
            for i in include:
                if i not in dirpath:
                    skip = True
                    break
            if skip:
                continue
            paths.append(dirpath)
            break
    return list(set(paths))

def preprocess(data, target_sum=10000, downsample=None, downsample_approach="nystroem"):
    from sklearn.preprocessing import PowerTransformer
    
    if target_sum is not None:
        # For each observation, normalize by the total count of all channels
        X_norm = target_sum * (data / data.sum(1)[:,None])  # rescale for floating point
    else:
        X_norm = data

    # power transform to a normal distribution
    X_norm = PowerTransformer().fit_transform(X_norm)
    
    # Downsample
    if downsample is not None:
        new_size = X_norm.shape[1]//downsample
        log(f"Reducing from {X_norm.shape[1]} to {new_size}", debug=debug)
        if downsample_approach == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=new_size)
            X_norm = reducer.fit_transform(X_norm)
        elif downsample_approach == "nystroem":
            from sklearn.kernel_approximation import Nystroem

            reducer = Nystroem(n_components=new_size)
            X_norm = reducer.fit_transform(X_norm)

    # z-score per feature
    X_norm = (X_norm - X_norm.mean(0)[None,:])/X_norm.std(0)[None,:]

    return X_norm
