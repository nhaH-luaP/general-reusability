import numpy as np
from torch.utils.data import Subset, DataLoader

def badge_query(model, budget, labeled_indices, unlabeled_indices, query_dset, device):
        query_loader = DataLoader(Subset(query_dset, unlabeled_indices), batch_size=128)
        grad_embedding = model.get_grad_representations(query_loader, device=device)
        rng = np.random.RandomState(42)
        chosen = kmeans_plusplus(grad_embedding.numpy(), budget, rng=rng)
        return [unlabeled_indices[idx] for idx in chosen]


def kmeans_plusplus(X, n_clusters, rng):
    # Start with highest grad norm since it is the "most uncertain"
    grad_norm = np.linalg.norm(X, ord=2, axis=1)
    idx = np.argmax(grad_norm)

    indices = [idx]
    centers = [X[idx]]
    dist_mat = []
    for _ in range(1, n_clusters):
        # Compute the distance of the last center to all samples
        dist = np.sqrt(np.sum((X - centers[-1])**2, axis=-1))
        dist_mat.append(dist)
        # Get the distance of each sample to its closest center
        min_dist = np.min(dist_mat, axis=0)
        min_dist_squared = min_dist**2
        if np.all(min_dist_squared == 0):
            raise ValueError('All distances to the centers are zero!')
        # sample idx with probability proportional to the squared distance
        p = min_dist_squared / np.sum(min_dist_squared)
        if np.any(p[indices] != 0):
            print('Already sampled centers have probability', p)
        idx = rng.choice(range(len(X)), p=p.squeeze())
        indices.append(idx)
        centers.append(X[idx])
    return indices