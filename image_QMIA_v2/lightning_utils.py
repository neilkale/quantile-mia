import os
import torch
import pytorch_lightning as pl

from optimizer_utils import build_optimizer
from scheduler_utils import build_scheduler
from train_utils import (
    gaussian_loss_fn,
    huber_gaussian_loss_fn,
    top_two_margin_score_fn,)

from timm.utils import accuracy
from torchmetrics.utilities.data import to_onehot
from torch_models import get_model
from pytorch_lightning.callbacks import BasePredictionWriter

### Utility functions for LightningBaseNet

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
        raise ValueError(
            "Batch should contain 3 elements: samples, targets, and base_samples"
        )
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

###

class LightningBaseNet(pl.LightningModule):
    def __init__(
        self,
        architecture,
        num_classes,
        base_image_size=-1,
        optimizer_params=None,
        loss_fn="cross_entropy",
        label_smoothing=0.0,
    ):
        super().__init__()

        if optimizer_params is None:
            optimizer_params = {}
        if loss_fn == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        self.optimizer_params = get_optimizer_params(optimizer_params)

        self.save_hyperparameters(
            "architecture",
            "num_classes",
            "base_image_size",
            "optimizer_params",
            "loss_fn"
        )
        self.model = get_model(architecture, num_classes, freeze_embedding=False)

        self.validation_step_outputs = []

    def forward(self, samples: torch.Tensor):
        logits = self.model(samples)
        return logits
    
    def training_step(self, batch, batch_idx: int):
        samples, targets, base_samples = get_batch(batch)
        logits = self.forward(base_samples)
        loss = self.loss_fn(logits, targets).mean()
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        self.log("ptl/loss", loss, on_epoch=True, prog_bar=True, on_step=False, sync_dist=True)
        self.log("ptl/acc1", acc1, on_epoch=True, prog_bar=True, on_step=False, sync_dist=True)
        self.log("ptl/acc5", acc5, on_epoch=True, prog_bar=True, on_step=False, sync_dist=True)

        return {
            "loss": loss,
            "acc1": acc1,
            "acc5": acc5,
        }
    
    def validation_step(self, batch, batch_idx: int):
        samples, targets, base_samples = get_batch(batch)

        logits = self.forward(base_samples)
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
        self.log("ptl/val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("ptl/val_acc1", avg_acc1, prog_bar=True, sync_dist=True)
        self.log("ptl/val_acc5", avg_acc5, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        samples, targets, base_samples = get_batch(batch)
        logits = self.forward(base_samples)
        loss = self.loss_fn(logits, targets)
        return logits, targets, loss

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

class LightningQMIA(pl.LightningModule):
    def __init__(
        self,
        architecture,
        base_architecture,
        image_size=-1,
        base_image_size=-1,
        hidden_dims=None,
        num_classes=10,
        base_model_dir=None,
        optimizer_params=None,
        loss_fn="gaussian",
        score_fn="top_two_margin",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_base_classes = num_classes

        print("Loading base model: {} from {}\nLoading attack model: {}".format(
            base_architecture, base_model_dir, architecture))

        # Get forward function of regression model
        model = get_model(
            architecture,
            2,
            False,
            hidden_dims=hidden_dims,
        )

        ## Create base model, load params from pickle, then define the scoring function and the logit embedding function
        base_model = get_model(
            base_architecture, self.num_base_classes, freeze_embedding=False
        )
        if base_model_dir is not None:
            base_state_dict = load_pickle(
                name="model.pickle",
                map_location=next(base_model.parameters()).device,
                base_model_dir=base_model_dir,
            )
            base_model.load_state_dict(base_state_dict)
        else:
            raise ValueError("Base model directory is not provided")

        self.model = model
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.optimizer_params = get_optimizer_params(optimizer_params)

        if loss_fn == "gaussian":
            self.loss_fn = gaussian_loss_fn
        elif loss_fn == "huber_gaussian":
            self.loss_fn = huber_gaussian_loss_fn
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        if score_fn == "top_two_margin":
            self.score_fn = top_two_margin_score_fn
        else:
            raise ValueError(f"Unknown score function: {score_fn}")
        
        self.validation_step_outputs = []

    def forward(self, samples: torch.Tensor, targets: torch.LongTensor = None, target_logits: torch.Tensor = None):
        """
        Forward pass through the model
        """
        scores = self.model(samples)
        return scores

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        samples, targets, base_samples = get_batch(batch)

        logits = self.base_model(base_samples)

        target_scores = self.score_fn(logits)
        pred_scores = self.forward(samples, targets, logits)

        loss = self.loss_fn(pred_scores, target_scores).mean()
        self.log("ptl/train_loss", loss, on_epoch=True, prog_bar=True, on_step=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx: int):
        samples, targets, base_samples = get_batch(batch)

        logits = self.base_model(base_samples)

        target_scores = self.score_fn(logits)
        pred_scores = self.forward(samples, targets, logits)

        loss = self.loss_fn(pred_scores, target_scores).mean()
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        rets = {
            "val_loss": loss,
            "base_acc1": acc1,
            "base_acc5": acc5,
        }
        self.validation_step_outputs.append(rets)
        return rets
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_acc1 = torch.stack(
            [x["base_acc1"] for x in self.validation_step_outputs]
        ).mean()
        avg_acc5 = torch.stack(
            [x["base_acc5"] for x in self.validation_step_outputs]
        ).mean()
        self.log("ptl/val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("ptl/base_acc1", avg_acc1, prog_bar=True, sync_dist=True)
        self.log("ptl/base_acc5", avg_acc5, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        samples, targets, base_samples = get_batch(batch)

        logits = self.base_model(base_samples)

        target_scores = self.score_fn(logits)
        pred_scores = self.forward(samples, targets, logits)

        loss = self.loss_fn(pred_scores, target_scores)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        
        return pred_scores, target_scores, logits, targets, loss
    
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
        

def load_pickle(name="model.pickle", map_location=None, base_model_dir=None):
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
