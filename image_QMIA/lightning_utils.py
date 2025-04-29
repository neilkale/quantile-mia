"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import os

import pytorch_lightning as pl
import torch
from optimizer_utils import build_optimizer
from pytorch_lightning.callbacks import BasePredictionWriter
from scheduler_utils import build_scheduler
from timm.utils import accuracy
from torch_models import get_model
from torchmetrics.utilities.data import to_onehot
from train_utils import (
    gaussian_loss_fn,
    label_logit_and_hinge_scoring_fn,
    pinball_loss_fn,
    rearrange_quantile_fn,
    top_two_margin_scoring_fn,
    margin_ratio_fn,
)

# base utilities


def get_optimizer_params(optimizer_params):
    "convenience function to add default options to optimizer params if not provided"
    # optimizer
    optimizer_params.setdefault("opt_type", "adamw")
    optimizer_params.setdefault("weight_decay", 0.0)
    optimizer_params.setdefault("lr", 1e-3)

    # scheduler
    optimizer_params.setdefault("scheduler", None)
    # optimizer_params.setdefault('min_factor', 1.)
    optimizer_params.setdefault("epochs", 100)  # needed for CosineAnnealingLR
    optimizer_params.setdefault("step_gamma", 0.1)  # decay fraction in step scheduler
    optimizer_params.setdefault(
        "step_fraction", 0.33
    )  # fraction of total epochs before step decay

    return optimizer_params


def get_batch(batch):
    if len(batch) == 2:
        samples, targets = batch
        base_samples = samples
    else:
        samples, targets, base_samples = batch
    return samples, targets, base_samples


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )


class LightningBaseNet(pl.LightningModule):
    def __init__(
        self,
        architecture,
        num_classes,
        image_size=-1,
        optimizer_params=None,
        loss_fn="Crossentropy",
        label_smoothing=0.0,
    ):
        super().__init__()
        if optimizer_params is None:
            optimizer_params = {}
        if loss_fn == "Crossentropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise NotImplementedError
        self.optimizer_params = get_optimizer_params(optimizer_params)

        self.save_hyperparameters(
            "architecture", "num_classes", "image_size", "optimizer_params", "loss_fn"
        )

        self.model = get_model(architecture, num_classes, freeze_embedding=False)

        self.validation_step_outputs = []

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        logits = self.model(samples)
        return logits

    def training_step(self, batch, batch_idx: int):
        samples, targets, base_samples = get_batch(batch)
        logits = self.forward(samples)
        loss = self.loss_fn(logits, targets).mean()
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        self.log("ptl/loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log("ptl/acc1", acc1, on_epoch=True, prog_bar=True, on_step=False)
        self.log("ptl/acc5", acc5, on_epoch=True, prog_bar=True, on_step=False)

        return {
            "loss": loss,
            "acc1": acc1,
            "acc5": acc5,
        }

    def validation_step(self, batch, batch_idx: int):
        samples, targets, base_samples = get_batch(batch)

        logits = self.forward(samples)
        loss = self.loss_fn(logits, targets).mean()
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        rets = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        self.validation_step_outputs.append(rets)
        return rets

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_acc1 = torch.stack(
            [x["val_acc1"] for x in self.validation_step_outputs]
        ).mean()
        avg_acc5 = torch.stack(
            [x["val_acc5"] for x in self.validation_step_outputs]
        ).mean()
        self.log("ptl/val_loss", avg_loss, prog_bar=True)
        self.log("ptl/val_acc1", avg_acc1, prog_bar=True)
        self.log("ptl/val_acc5", avg_acc5, prog_bar=True)
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        samples, targets, base_samples = get_batch(batch)

        logits = self.forward(samples)
        loss = self.loss_fn(logits, targets)
        # get hinge score
        oh_label = to_onehot(targets, logits.shape[-1]).bool()
        score = logits[oh_label]
        score -= torch.max(logits[~oh_label].view(logits.shape[0], -1), dim=1)[0]
        return logits, targets, loss, score
        # return score

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.model,
            opt_type=self.optimizer_params["opt_type"],
            lr=self.optimizer_params["lr"],
            weight_decay=self.optimizer_params["weight_decay"],
        )
        interval = "epoch"
        lr_scheduler = build_scheduler(
            scheduler=self.optimizer_params["scheduler"],
            epochs=self.optimizer_params["epochs"],
            # min_factor=self.optimizer_params['min_factor'],
            optimizer=optimizer,
            mode="max",
            step_gamma=self.optimizer_params["step_gamma"],
            lr=self.optimizer_params["lr"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": interval,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "ptl/val_acc1",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            },
        }


# Lightning wrapper for MIA/QR model
class LightningQMIA(pl.LightningModule):
    def __init__(
        self,
        architecture,
        base_architecture,
        num_base_classes,
        image_size,
        hidden_dims,
        freeze_embedding,
        # base_model_name_prefix,
        low_quantile,
        high_quantile,
        n_quantile,
        # cumulative_qr,
        optimizer_params,
        base_model_path=None,
        rearrange_on_predict=True,
        use_target_label=False,
        use_hinge_score=False,
        use_logscale=False,
        use_gaussian=False,
        return_mean_logstd=False,
        use_target_dependent_scoring=False,
        use_target_inputs=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_target_dependent_scoring = use_target_dependent_scoring
        assert not (
            use_target_dependent_scoring and use_target_inputs
        ), "target_dependent scoring should not be used with use_target_inputs"

        self.use_target_inputs = use_target_inputs
        self.num_base_classes = num_base_classes
        self.base_n_outputs = 2 if use_gaussian else n_quantile
        if self.use_target_dependent_scoring:
            n_outputs = self.base_n_outputs * self.num_base_classes
        else:
            n_outputs = self.base_n_outputs

        model, base_model = model_setup(
            architecture=architecture,
            base_architecture=base_architecture,
            image_size=image_size,
            num_quantiles=n_outputs,
            num_base_classes=num_base_classes,
            # base_model_name_prefix=base_model_name_prefix,
            hidden_dims=hidden_dims,
            freeze_embedding=freeze_embedding,
            base_model_path=base_model_path,
            extra_inputs=6 if self.use_target_inputs else None, # MODIFIED TO USE # OF SUMMARY STATISTICS INSTEAD OF `num_base_classes`
        )

        self.model = model
        self.base_model = base_model
        self.base_model_path = base_model_path
        self.use_gaussian = use_gaussian
        self.return_mean_logstd = return_mean_logstd

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        if use_logscale:
            self.QUANTILE = torch.sort(
                1
                - torch.logspace(
                    low_quantile, high_quantile, n_quantile, requires_grad=False
                )
            )[0].reshape([1, -1])
        else:
            self.QUANTILE = torch.sort(
                torch.linspace(
                    low_quantile, high_quantile, n_quantile, requires_grad=False
                )
            )[0].reshape([1, -1])

        if self.use_gaussian:
            self.loss_fn = gaussian_loss_fn
            if use_target_label:
                self.target_scoring_fn = label_logit_and_hinge_scoring_fn
            else:
                self.target_scoring_fn = top_two_margin_scoring_fn
            self.rearrange_on_predict = False
        else:
            self.loss_fn = pinball_loss_fn
            if use_target_label:
                self.target_scoring_fn = label_logit_and_hinge_scoring_fn
                if not use_hinge_score:
                    raise NotImplementedError
            else:
                self.target_scoring_fn = top_two_margin_scoring_fn
            self.rearrange_on_predict = rearrange_on_predict and not use_logscale

        optimizer_params.update(**kwargs)
        self.optimizer_params = get_optimizer_params(optimizer_params)

        self.validation_step_outputs = []

    def forward(
        self, samples: torch.Tensor, targets: torch.LongTensor = None, target_logits: torch.Tensor = None
    ) -> torch.Tensor:
        if self.use_target_inputs:
            # oh_targets = to_onehot(targets, self.num_base_classes)
            # oh_targets = torch.sort(target_logits, dim=-1, descending=True)[0]
            sorted_logits = torch.sort(target_logits, dim=1, descending=True)[0]
            softmax_probs = torch.softmax(target_logits, dim=1)
            entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1)
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]
            mean = torch.mean(target_logits, dim=1, keepdim=True)
            std = torch.std(target_logits, dim=1, keepdim=True) + 1e-10  # prevent div by zero
            skewness = torch.mean(((target_logits - mean) / std) ** 3, dim=1, keepdim=True)
            kurtosis = torch.mean(((target_logits - mean) / std) ** 4, dim=1, keepdim=True) - 3
            oh_targets = torch.cat(
                [
                    entropy.unsqueeze(1),
                    margin.unsqueeze(1),
                    mean,
                    std,
                    skewness,
                    kurtosis,
                ],
                dim=1,
            )
            scores = self.model(samples, oh_targets)
            return scores
        scores = self.model(samples)
        if self.use_target_dependent_scoring:
            oh_targets = to_onehot(targets, self.num_base_classes).unsqueeze(1)
            scores = (
                scores.reshape(
                    [
                        -1,
                        self.base_n_outputs,
                        self.num_base_classes,
                    ]
                )
                * oh_targets
            ).sum(-1)
        return scores

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        samples, targets, base_samples = get_batch(batch)

        target_score, target_logits = self.target_scoring_fn(
            base_samples, targets, self.base_model
        )
        scores = self.forward(samples, targets, target_logits)

        loss = self.loss_fn(
            scores, target_score, self.QUANTILE.to(scores.device)
        ).mean()
        self.log("ptl/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        samples, targets, base_samples = get_batch(batch)
        # print('VALIDATION STEP', self.model.training), print(self.base_model.training)
        target_score, target_logits = self.target_scoring_fn(
            base_samples, targets, self.base_model
        )

        scores = self.forward(samples, targets, target_logits)
        if self.rearrange_on_predict and not self.use_gaussian:
            scores = rearrange_quantile_fn(
                scores, self.QUANTILE.to(scores.device).flatten()
            )
        loss = self.loss_fn(
            scores, target_score, self.QUANTILE.to(scores.device)
        ).mean()

        rets = {
            "val_loss": loss,
            "scores": scores,
            "targets": target_score,
        }
        self.validation_step_outputs.append(rets)
        return rets

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        targets = torch.concatenate(
            [x["targets"] for x in self.validation_step_outputs], dim=0
        )
        scores = torch.concatenate(
            [x["scores"] for x in self.validation_step_outputs], dim=0
        )

        self.validation_step_outputs.clear()  # free memory
        self.log("ptl/val_loss", avg_loss, sync_dist=True, prog_bar=True)

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     samples, targets, base_samples = get_batch(batch)

    #     target_score, target_logits = self.target_scoring_fn(
    #         base_samples, targets, self.base_model
    #     )

    #     scores = self.forward(samples, targets, target_logits)
        # if self.rearrange_on_predict and not self.use_gaussian:
        #     scores = rearrange_quantile_fn(
        #         scores, self.QUANTILE.to(scores.device).flatten()
        #     )
        # loss = self.loss_fn(scores, target_score, self.QUANTILE.to(scores.device))
        # base_acc1, base_acc5 = per_sample_accuracy(target_logits, targets, topk=(1, 5))

    #     if self.use_gaussian and not self.return_mean_logstd:
    #         # use torch distribution to output quantiles
    #         mu = scores[:, 0]
    #         log_std = scores[:, 1]
    #         std = torch.exp(log_std)
    #
    #         scores = mu.reshape([-1, 1]) + torch.exp(log_std).reshape(
    #             [-1, 1]
    #         ) * torch.erfinv(2 * self.QUANTILE.to(scores.device) - 1).reshape(
    #             [1, -1]
    #         ) * math.sqrt(
    #             2
    #         )
    #         assert (
    #             scores.ndim == 2
    #             and scores.shape[0] == targets.shape[0]
    #             and scores.shape[1] == self.QUANTILE.shape[1]
    #         ), "inverse cdf quantiles have the wrong shape, got {} {} {}".format(
    #             scores.shape, targets.shape, self.QUANTILE.size()
    #         )

    #     return  scores, target_score, loss, base_acc1, base_acc5, targets, # target_probs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        samples, targets, base_samples = get_batch(batch)

        # 1) your existing forward‐and‐loss logic
        target_score, target_logits = self.target_scoring_fn(
            base_samples, targets, self.base_model
        )
        scores = self.forward(samples, targets, target_logits)
        # … your existing rearrange_on_predict / loss_fn / base_acc1/5 code …
        if self.rearrange_on_predict and not self.use_gaussian:
            scores = rearrange_quantile_fn(
                scores, self.QUANTILE.to(scores.device).flatten()
            )
        loss = self.loss_fn(scores, target_score, self.QUANTILE.to(scores.device))
        base_acc1, base_acc5 = per_sample_accuracy(target_logits, targets, topk=(1, 5))

        # 2) NEW: compute p‐values
        if self.use_gaussian:
            # your mu/log_std branch
            mu      = scores[:, 0]
            log_std = scores[:, 1]
            std     = torch.exp(log_std)
            dist    = torch.distributions.Normal(mu, std)
            # p_values = dist.cdf(target_score)               # shape [batch]
            z_scores = (target_score - mu) / std
        else:
            # non‐Gaussian: invert your Q predicted quantiles
            # scores: [batch, Q],  self.QUANTILE: [1, Q] sorted ascending
            Qs    = self.QUANTILE.to(scores.device).flatten()  # [Q]
            # make sure 'scores' are sorted along dim=1
            sorted_scores, _ = torch.sort(scores, dim=1)       # [batch, Q]
            # for each i, find the smallest index j where sorted_scores[i,j] >= target_score[i]
            idxs = torch.searchsorted(sorted_scores, target_score.unsqueeze(1))
            idxs = idxs.clamp(0, sorted_scores.size(1)-1).squeeze(1)  # [batch]
            p_values = Qs[idxs]                                  # [batch]

        # 3) return it!
        return scores, target_score, loss, base_acc1, base_acc5, targets, z_scores, target_logits

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.model,
            opt_type=self.optimizer_params["opt_type"],
            lr=self.optimizer_params["lr"],
            weight_decay=self.optimizer_params["weight_decay"],
        )
        interval = "epoch"

        lr_scheduler = build_scheduler(
            scheduler=self.optimizer_params["scheduler"],
            epochs=self.optimizer_params["epochs"],
            step_fraction=self.optimizer_params["step_fraction"],
            step_gamma=self.optimizer_params["step_gamma"],
            optimizer=optimizer,
            mode="min",
            lr=self.optimizer_params["lr"],
        )
        opt_and_scheduler_config = {
            "optimizer": optimizer,
        }
        if lr_scheduler is not None:
            opt_and_scheduler_config["lr_scheduler"] = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                "interval": interval,
                "frequency": 1,
                "monitor": "ptl/val_loss",
                "strict": True,
                "name": None,
            }

        return opt_and_scheduler_config


# Convenience function to create models and potentially load weights for base classifier
def model_setup(
    architecture,
    base_architecture,
    image_size,
    num_quantiles,
    num_base_classes,
    hidden_dims,
    freeze_embedding,
    base_model_path=None,
    extra_inputs=None,
):
    # Get forward function of regression model
    model = get_model(
        architecture,
        num_quantiles,
        image_size,
        freeze_embedding,
        hidden_dims=hidden_dims,
        extra_inputs=extra_inputs,
    )

    ## Create base model, load params from pickle, then define the scoring function and the logit embedding function
    base_model = get_model(
        base_architecture, num_base_classes, image_size, freeze_embedding=False
    )
    if base_model_path is not None:
        base_state_dict = load_pickle(
            name="model.pickle",
            map_location=next(base_model.parameters()).device,
            base_model_dir=os.path.dirname(base_model_path),
        )
        base_model.load_state_dict(base_state_dict)

    return model, base_model


def load_pickle(name="quantile_model.pickle", map_location=None, base_model_dir=None):
    # pickle_path = os.path.join(args.log_root, args.dataset, name.replace('/', '_'))
    pickle_path = os.path.join(base_model_dir, name.replace("/", "_"))
    if map_location:
        state_dict = torch.load(pickle_path, map_location=map_location)
    else:
        state_dict = torch.load(pickle_path)
    return state_dict

def per_sample_accuracy(output, target, topk=(1,)):
    """
    Computes per-sample accuracy over the k top predictions
    Returns a list of tensors, each of size [batch_size], with 1.0 for correct predictions and 0.0 for incorrect ones
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Reshape to [k, batch_size] and check if any of top-k predictions are correct for each sample
            correct_k = correct[:k].view(k, batch_size)
            per_sample_correct = correct_k.any(dim=0).float()  # [batch_size] tensor with 1.0/0.0 values
            res.append(per_sample_correct)

        return res
