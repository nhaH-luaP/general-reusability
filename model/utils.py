import torch

@torch.inference_mode()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    if target.ndim == 2:
        target = target.max(dim=1)[1]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        res.append(correct_k * (100.0 / batch_size))
    return res

def train_one_epoch(labeled_trainloader, model, optimizer, criterion, device='cuda'):
    model.to(device)
    model.train()
    n_samples, n_correct, total_loss = 0, 0, 0
    
    for X_batch, y_batch in labeled_trainloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)
        current_batch_size = X_batch.shape[0]
        loss = criterion(out, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_samples += current_batch_size
        n_correct += torch.sum(torch.eq(torch.argmax(out.softmax(-1), -1), y_batch)).item()
        total_loss += loss.item() * current_batch_size

    return {
        "acc1": 0 if n_samples == 0 else n_correct/n_samples*100,
        "total_loss": 0 if n_samples == 0 else total_loss/n_samples
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device='cuda'):
    model.eval()
    model.to(device)

    # Forward prop in distribution
    logits_id, targets_id, = [], []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Model specific test loss and accuracy for in domain testset
    acc1 = accuracy(logits_id, targets_id, (1,))[0].item()
    loss = criterion(logits_id, targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()

    return {
        "acc1": acc1,
        "total_loss": loss,
        "nll": nll
    }