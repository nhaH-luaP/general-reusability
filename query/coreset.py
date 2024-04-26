from torch.utils.data import Subset, DataLoader
import torch

def coreset_query(model, budget, labeled_indices, unlabeled_indices, query_dset, device):
    unlabeled_dataloader = DataLoader(Subset(query_dset, unlabeled_indices))
    labeled_dataloader = DataLoader(Subset(query_dset, labeled_indices))

    features_unlabeled = model.get_representations(unlabeled_dataloader, device)
    features_labeled = model.get_representations(labeled_dataloader, device)

    chosen = kcenter_greedy(features_unlabeled, features_labeled, budget)
    return [unlabeled_indices[idx] for idx in chosen]

def kcenter_greedy(features_unlabeled, features_labeled, acq_size):
    n_unlabeled = len(features_unlabeled)

    distances = torch.cdist(features_unlabeled, features_labeled)
    min_dist, _ = torch.min(distances, axis=1)

    idxs = []
    for _ in range(acq_size):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = torch.cdist(features_unlabeled, features_unlabeled[idx].unsqueeze(0))
        for j in range(n_unlabeled):
            min_dist[j] = torch.min(min_dist[j], dist_new_ctr[j, 0])
    return idxs