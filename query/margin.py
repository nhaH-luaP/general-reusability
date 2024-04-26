import torch
from torch.utils.data import Subset, DataLoader

def margin_query(model, budget, labeled_indices, unlabeled_indices, query_dset, device):
    query_loader = DataLoader(Subset(query_dset, unlabeled_indices), batch_size=128)
    logits = model.get_logits(dataloader=query_loader, device=device)
    probas = logits.softmax(dim=-1)
    top2_probas = torch.topk(probas, k=2, dim=-1)[0]
    diff = torch.subtract(top2_probas[:,0], top2_probas[:,1])
    indices = torch.topk(diff, k=budget, largest=False)[1].tolist()

    # Indices are now the indices with the lowest marginal certainty with regards to unlabeled_indices, but not
    # the actual dataset. Therefore we need to transform them accordingly
    real_indices = [unlabeled_indices[i] for i in indices]
    return real_indices