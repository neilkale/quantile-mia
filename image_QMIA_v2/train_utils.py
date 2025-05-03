import torch

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

###########
# Score functions for the attack network
###########

def top_two_margin_score_fn(logits):
    # z_{max}(x) - z_{second-max}(x)
    top_two = torch.topk(logits, 2, dim=1)
    score = top_two.values[:, 0] - top_two.values[:, 1]
    return score