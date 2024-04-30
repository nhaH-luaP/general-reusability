import torch
import torch.nn as nn
import torch.nn.functional as F


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model = model.to(device)
    model = model.train()
    
    total_loss, n_samples = 0, 0
    for views, _ in dataloader:
        views = torch.cat(views, dim=0)
        views = views.to(device)
        out = model(views)
        loss = criterion(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * out.shape[0]
        n_samples += out.shape[0]

    return {'loss':total_loss/n_samples}



def evaluate(model, dataloader, criterion, device):
    model = model.to(device)
    total_loss, n_samples = 0, 0
    for views, _ in dataloader:
        views = torch.cat(views, dim=0)
        views = views.to(device)
        out = model(views)
        loss = criterion(out)
        total_loss += loss.item() * out.shape[0]
        n_samples += out.shape[0]

    return {'loss':total_loss/n_samples}



class RepeatTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]



class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        assert temperature > 0.0, "The temperature must be a positive float!"
        self.temperature = temperature

    def forward(self, batch):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(batch[:, None, :], batch[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        infoNCE = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        infoNCE = infoNCE.mean()

        return infoNCE