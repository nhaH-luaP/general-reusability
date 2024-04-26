import torch
from torch.utils.data import Subset, DataLoader

def bait_query(model, budget, labeled_indices, unlabeled_indices, query_dset, device, expectation_topk=None,
                 normalize_top_probas=True,
                 fisher_approximation='full',
                 num_grad_samples=None,
                 grad_likelihood='cross_entropy',
                 grad_selection='magnitude',
                 select='forward_backward',
                 fisher_batch_size=32,):
    # TODO: Discuss with Denis if this makes sense or just further complicates the whole repository
    return None