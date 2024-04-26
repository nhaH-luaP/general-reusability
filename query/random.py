import random

def random_query(model, budget, labeled_indices, unlabeled_indices, query_dset, device):
    newly_labeled_indices = random.sample(unlabeled_indices, budget)
    return newly_labeled_indices