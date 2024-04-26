import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    # Calculating NearestNeighbors using sklearn
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # 0 index is the same sample, dropping it
    distances, indices = distances[:, 1:], indices[:, 1:]
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, n_init='auto')
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000,  n_init='auto')
        km.fit_predict(features)
    return km.labels_


def typiclust_query(model, budget, labeled_indices, unlabeled_indices, query_dset, device):
    MIN_CLUSTER_SIZE, MAX_NUM_CLUSTERS, K_NN = 5, 500, 20
    num_clusters = min(len(labeled_indices) + budget, MAX_NUM_CLUSTERS)

    dataloader = DataLoader(query_dset, batch_size=128, shuffle=False)

    features = model.get_representations(dataloader, device)
    labels = kmeans(features, num_clusters=num_clusters)
    existing_indices = np.array(labeled_indices)

    # Counting cluster sizes and number of labeled samples per cluster
    cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
    clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                'neg_cluster_size': -1 * cluster_sizes})
    
    # Drop too small clusters
    clusters_df = clusters_df[clusters_df.cluster_size > MIN_CLUSTER_SIZE]

    # Sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
    clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
    labels[existing_indices] = -1

    selected = []

    for i in range(budget):
        cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
        indices = (labels == cluster).nonzero()[0]
        rel_feats = features[indices]

        # In case we have too small cluster, calculate density among half of the cluster
        typicality = calculate_typicality(rel_feats, min(K_NN, len(indices) // 2))
        idx = indices[typicality.argmax()]
        selected.append(idx)
        labels[idx] = -1

    # Transform to 32int to be saveable for JSON dump and check for errors
    selected = [int(i) for i in selected]
    selected_array = np.array(selected)
    assert len(selected) == budget, 'added a different number of samples'
    assert len(np.intersect1d(selected_array, existing_indices)) == 0, 'should be new samples'

    return selected