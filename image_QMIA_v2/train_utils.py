import torch
import torch.nn.functional as F

###########
# distribution learning losses
###########

def pinball_loss_fn(score, target, quantile):
    target = target.reshape([-1, 1])
    delta_score = target - score
    loss = torch.nn.functional.relu(delta_score) * quantile + torch.nn.functional.relu(
        -delta_score
    ) * (1.0 - quantile)
    return loss


def gaussian_loss_fn(score, target):
    mu = score[:, 0]
    log_std = score[:, 1]
    loss = log_std + 0.5 * torch.exp(-2 * log_std) * (target - mu) ** 2
    return loss


def huber_gaussian_loss_fn(score, target, delta=1.0):
    mu = score[:, 0]
    log_std = score[:, 1]
    sigma = torch.exp(log_std)
    huber_loss = F.huber_loss(mu, target, reduction='none', delta=delta)
    loss = log_std + (huber_loss / (sigma ** 2))
    return loss

###########
# Score functions for the attack network
###########

def top_two_margin_score_fn(logits):
    # z_{max}(x) - z_{second-max}(x)
    top_two = torch.topk(logits, 2, dim=1)
    score = top_two.values[:, 0] - top_two.values[:, 1]
    return score

def true_margin_score_fn(logits, target):
    # z_{target}(x) - z_{second-max}(x)
    top_two = torch.topk(logits, 2, dim=1)
    target_idx = target.view(-1, 1)
    target_score = torch.gather(top_two.values, 1, target_idx)
    second_max_score = top_two.values[:, 1]
    score = target_score - second_max_score
    return score