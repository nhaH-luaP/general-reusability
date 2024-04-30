import numpy as np
import logging

import torch.nn.functional as F
import torch


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class SemiLoss(object):
    def __init__(self, lambda_u, rampup_length):
        self.lambda_u = lambda_u
        self.rampup_length = rampup_length

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.lambda_u * linear_rampup(epoch, self.rampup_length)


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def train_one_epoch_mixmatch(labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch, n_train_iterations, device='cuda', T=1.0, alpha=0.9, print_freq=None, lr_scheduler=None):
    model.train()
    model = model.to(device)

    total_loss, total_loss_x, total_loss_u, n_samples, n_correct = 0, 0, 0, 0, 0

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    for batch_idx in range(n_train_iterations):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2), _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = next(unlabeled_train_iter)


        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x_one_hot = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)


        inputs_x, targets_x, targets_x_one_hot = inputs_x.to(device), targets_x.to(device), targets_x_one_hot.to(device)
        inputs_u = inputs_u.to(device)
        inputs_u2 = inputs_u2.to(device)

        # Compute pseudo-labels of unlabeled samples
        with torch.no_grad():
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # Perform mixup between all available data
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x_one_hot, targets_u, targets_u], dim=0)

        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # Interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        # Run them through the model
        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # Put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        # Calculate loss
        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/n_train_iterations)
        loss = Lx + w * Lu

        # Record loss
        total_loss += loss.item() * batch_size
        total_loss_x += Lx.item() * batch_size
        total_loss_u += Lu.item() * batch_size

        # Record accuracy on labeled data outside of mixup
        predictions = logits_x.softmax(dim=-1).argmax(dim=-1)
        n_correct += torch.sum(predictions == targets_x).item()
        n_samples += batch_size

        # Update Parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        # Logging inbetween
        if print_freq and batch_idx % print_freq == 0:
            logging.info(f"[Batch {batch_idx}/{n_train_iterations}] Total Loss: {loss}    Supervised Loss: {Lx}    Unsupervised Loss {Lu}")

    return {
        "acc1":n_correct/n_samples*100,
        "total_loss":total_loss/n_samples,
        "supervised_loss":total_loss_x/n_samples,
        "unsupervised_loss":total_loss_u/n_samples
    }