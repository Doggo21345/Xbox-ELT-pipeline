import os 
import pandas as pd 
import pickle 
from databricks import sql 
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import umap 
import hdbscan
def train_n_optimize():
    param_grid = {
        'n_neighbors': [15,30],
        'min_dist':[0,.1],
        'min_cluster_size':[20,50],
        'min_samples':[1,5]
    }
    results = []
    for params in ParameterGrid(param_grid):
        reducer = umap.UMAP(
            n_neighbors=params['n_neighbors'], 
            min_dist = params['min_dist'],
            random_state = 42 
        )
        clusterer = hdbscan.HDBSCAN (
            min_cluster_size=params['n_neighbors'],
            min_samples = params['min_dist'],
            gen_min_span_tree=True
        )
        embedding = reducer.fit_transform(X_final)
        labels = clusterer.fit_predict(embedding)
        mask = labels != -1 
        if mask.sum() > 1 and len(set(labels(mask))) > 1:
            sil = silhouette_score(embedding[mask],labels[mask])
        else:
            sil = -1
        results.append ({
            'silhoute': sil, 
            'params': params, 
            'model': (reducer, clusterer)
        })
    best = max(results,key = lambda x:x['silhouette'])
    if best['silhouette'] < .60:
        raise ValueError(f"Model health check failed! Best score: {best['silhouette']}")
        
    return best['model'] # Returns tuple of (reducer, clusterer)


