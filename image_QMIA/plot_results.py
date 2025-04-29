import os
import shutil
import warnings

warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import math
from glob import glob
import time

import numpy as np
import pytorch_lightning as pl
import torch
from torch import distributed as dist
from analysis_utils import plot_performance_curves, plot_performance_curves_id_ood
from data_utils import CustomDataModule
from lightning_utils import CustomWriter, LightningQMIA
from train_mia_ray import argparser
from tqdm import tqdm

import atexit
@atexit.register
def _cleanup_cuda():
    # this runs when Python exits (either normally or via unhandled exception)
    torch.cuda.empty_cache()

def plot_model(
    args,
    checkpoint_path,
    fig_name="best",
    recompute_predictions=True,
    return_mean_logstd=False,
):
    # -------------------------
    # 1) Setup prediction folder
    # -------------------------
    if return_mean_logstd:
        fig_name = f"raw_{fig_name}"
        prediction_output_dir = os.path.join(
            args.root_checkpoint_path,
            "raw_predictions",
            fig_name,
        )
    else:
        # handle leading slash in root_checkpoint_path
        base = (
            args.root_checkpoint_path[1:]
            if args.root_checkpoint_path.startswith("/")
            else args.root_checkpoint_path
        )
        prediction_output_dir = os.path.join(
            base,
            "predictions",
            fig_name,
        )

    print("Saving predictions to", prediction_output_dir)
    os.makedirs(prediction_output_dir, exist_ok=True)

    # -------------------------
    # 2) Build model_target_quantiles tensor
    # -------------------------
    if args.use_log_quantile:
        q = 1.0 - torch.logspace(
            args.low_quantile, args.high_quantile, args.n_quantile, requires_grad=False
        )
    else:
        q = 1.0 - torch.linspace(
            args.low_quantile, args.high_quantile, args.n_quantile, requires_grad=False
        )

    model_target_quantiles = torch.sort(q)[0].unsqueeze(0)  # shape [1, Q]

    # -------------------------
    # 3) Possibly rerun Trainer.predict
    # -------------------------
    needs_run = (
        recompute_predictions
        or len(glob(os.path.join(prediction_output_dir, "*.pt"))) == 0
    )
    if needs_run:
        # 2) Only rank 0 should delete the old preds folder
        is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
        if is_rank0:
            shutil.rmtree(prediction_output_dir, ignore_errors=True)
            os.makedirs(prediction_output_dir, exist_ok=True)

        # 4) Build data+model+trainer
        datamodule = CustomDataModule(
            dataset_name=args.base_dataset,
            mode="eval",
            num_workers=7,
            image_size=args.image_size,
            batch_size=args.batch_size,
            data_root=args.data_root,
        )
        
        print(f"[rank 0] reloading from {checkpoint_path}")
        lightning_model = LightningQMIA.load_from_checkpoint(checkpoint_path)
        lightning_model.QUANTILE = model_target_quantiles
        if return_mean_logstd:
            lightning_model.return_mean_logstd = True

        pred_writer = CustomWriter(
            output_dir=prediction_output_dir,
            write_interval="epoch",
        )
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto" if torch.cuda.is_available() else "cpu",
            callbacks=[pred_writer],
            devices=-1 if torch.cuda.is_available() else 1,
            enable_progress_bar=True,
        )

        # 5) Everybody runs predict (so each rank writes its chunk via CustomWriter)
        trainer.predict(
            lightning_model,
            datamodule,
            return_predictions=False,
        )

# -------------------------
    # 4) Load & merge saved .pt files on rank-0
    # -------------------------
    print(f"[plot_model] Loading predictions from {prediction_output_dir}", flush=True)
    pt_paths = sorted(glob(os.path.join(prediction_output_dir, "*.pt")))
    print(f"[plot_model] Found {len(pt_paths)} .pt files", flush=True)

    # 4a) Load each chunk (a list of N_splits lists of tuples)
    loaded_chunks = []
    for i, p in enumerate(pt_paths, start=1):
        print(f"[plot_model]   Loading file {i}/{len(pt_paths)}: {p}", flush=True)
        loaded_chunks.append(torch.load(p, map_location="cpu"))

    # 4b) Figure out how many “splits” (dataloaders) we have per chunk
    n_splits = len(loaded_chunks[0])
    print(f"[plot_model] Detected {n_splits} splits per chunk; merging …", flush=True)

    # 4c) Merge across ranks for each split
    merged_batches = [[] for _ in range(n_splits)]
    for chunk in loaded_chunks:
        for split_idx in range(n_splits):
            print(f"[plot_model]   Merging chunk into split {split_idx}", flush=True)
            # chunk[split_idx] is a list-of-tuples
            merged_batches[split_idx].extend(chunk[split_idx])

    # 4d) Now join each split’s list-of-tuples into a list of Tensors
    def join_list_of_tuples(list_of_tuples):
        num_fields = len(list_of_tuples[0])
        out = []
        for fld in range(num_fields):
            # collect field fld across every tuple
            elems = [tup[fld] for tup in list_of_tuples]
            try:
                out.append(torch.concat(elems, dim=0))
            except:
                out.append(torch.tensor(elems))
        return out

    split_fields = []
    for split_idx in range(n_splits):
        fields = join_list_of_tuples(merged_batches[split_idx])
        shapes = [tuple(t.shape) for t in fields]
        print(f"[plot_model]  Split {split_idx} → {len(fields)} fields, shapes: {shapes}", flush=True)
        split_fields.append(fields)

    # 4e) Pick which splits you actually want as “private” and “test”
    # Here I’m assuming split_fields[0] == private, split_fields[1] == test
    private_fields = split_fields[-1]
    test_fields    = split_fields[1]

    if n_splits > 2:
        extras = list(range(2, n_splits))
        print(f"[plot_model]  Warning: ignoring extra splits {extras}", flush=True)

    # 4f) Unpack
    (
        private_predicted_quantile_threshold,
        private_target_score,
        private_loss,
        private_base_acc1,
        private_base_acc5,
        private_targets,
        private_p_values,
        _
    ) = private_fields

    (
        test_predicted_quantile_threshold,
        test_target_score,
        test_loss,
        test_base_acc1,
        test_base_acc5,
        test_targets,
        test_p_values,
        _
    ) = test_fields

    print(f"[plot_model] Final tensor shapes:\n"
          f"   private_predicted_quantile_threshold: {private_predicted_quantile_threshold.shape}\n"
          f"   private_p_values:                     {private_p_values.shape}\n"
          f"   test_predicted_quantile_threshold:    {test_predicted_quantile_threshold.shape}\n"
          f"   test_p_values:                        {test_p_values.shape}",
          flush=True)

    model_target_quantiles = model_target_quantiles.cpu().numpy().reshape(-1)

    # if return_mean_logstd:
    #     # model_target_quantiles = model_target_quantiles[1:-1]
    #     dislocated_quantiles = torch.erfinv(
    #         2 * torch.Tensor(model_target_quantiles) - 1
    #     ).reshape([1, -1]) * math.sqrt(2)

    #     public_mu = test_predicted_quantile_threshold[:, 0].reshape([-1, 1])
    #     public_std = torch.exp(test_predicted_quantile_threshold[:, 1]).reshape([-1, 1])
    #     test_predicted_quantile_threshold = (
    #         public_mu + public_std * dislocated_quantiles
    #     )

    #     private_mu = private_predicted_quantile_threshold[:, 0].reshape([-1, 1])
    #     private_std = torch.exp(private_predicted_quantile_threshold[:, 1]).reshape(
    #         [-1, 1]
    #     )
    #     private_predicted_quantile_threshold = (
    #         private_mu + private_std * dislocated_quantiles
    #     )
    
    drop_set = set(args.cls_drop or [])  
    # curiosity_set = [0] # For examining performance on a class if we don't drop it.
    # drop_set |= set(curiosity_set)

    if len(drop_set) > 0:

        priv_ood   = torch.tensor([l.item() in drop_set for l in private_targets])
        test_ood   = torch.tensor([l.item() in drop_set for l in test_targets])
        priv_id    = ~priv_ood
        test_id    = ~test_ood

        def pct(x): return 100*x.mean().item()
        print("Model accuracy on training set  ID: {:.2f}%  OOD: {:.2f}%".format(
            pct(private_base_acc1[priv_id]), pct(private_base_acc1[priv_ood])))
        print("Model accuracy on test set     ID: {:.2f}%  OOD: {:.2f}%".format(
            pct(test_base_acc1[test_id]),    pct(test_base_acc1[test_ood])))
        
        if args.base_dataset:
            dset, split = args.dataset.split("/")
            base_dset, base_split = args.base_dataset.split("/")
            dset_dir = os.path.join(base_dset + "_" + dset, base_split + "_" + split)
        else:
            dset_dir = args.dataset

        # Use the new side-by-side plotting function
        plot_performance_curves_id_ood(
            np.asarray(private_target_score),
            np.asarray(test_target_score),
            private_predicted_score_thresholds=np.asarray(
                private_predicted_quantile_threshold
            ),
            public_predicted_score_thresholds=np.asarray(test_predicted_quantile_threshold),
            model_target_quantiles=model_target_quantiles,
            private_p_values=np.asarray(private_p_values),
            public_p_values =np.asarray(test_p_values),
            model_name="Quantile Regression",
            private_id_mask=priv_id,
            private_ood_mask=priv_ood,
            public_id_mask=test_id,
            public_ood_mask=test_ood,
            use_logscale=True,
            fontsize=12,
            savefig_path="./plots/{}/{}/{}/{}/ray/use_hinge_{}/use_target_label_{}/use_target_inputs_{}/cls_drop_{}/{}.png".format(
                args.model_name_prefix,
                dset_dir,
                args.base_architecture.replace("/", "_"),
                args.architecture[1:].replace("/", "_") if args.architecture.startswith("/") else args.architecture.replace("/", "_"),
                args.use_hinge_score,
                args.use_target_label,
                args.use_target_inputs,
                "".join(str(c) for c in args.cls_drop),
                fig_name,
            ),
        )
    else:
        print(
            "Model accuracy on training set {:.2f}%".format(
                np.mean(private_base_acc1.numpy())
            )
        )

        if args.base_dataset:
            dset, split = args.dataset.split("/")
            base_dset, base_split = args.base_dataset.split("/")
            dset_dir = os.path.join(base_dset + "_" + dset, base_split + "_" + split)
        else:
            dset_dir = args.dataset

        print("Model accuracy on test set  {:.2f}%".format(np.mean(test_base_acc1.numpy())))
        plot_result = plot_performance_curves(
            np.asarray(private_target_score),
            np.asarray(test_target_score),
            private_predicted_score_thresholds=np.asarray(
                private_predicted_quantile_threshold
            ),
            public_predicted_score_thresholds=np.asarray(test_predicted_quantile_threshold),
            private_p_values=np.asarray(private_p_values),
            public_p_values =np.asarray(test_p_values),
            model_target_quantiles=model_target_quantiles,
            model_name="Quantile Regression",
            use_logscale=True,
            fontsize=12,
            savefig_path="./plots/{}/{}/{}/{}/ray/use_hinge_{}/use_target_label_{}/use_target_inputs_{}/cls_drop_{}/{}.png".format(
                args.model_name_prefix,
                dset_dir,
                args.base_architecture.replace("/", "_"),
                args.architecture[1:].replace("/", "_") if args.architecture.startswith("/") else args.architecture.replace("/", "_"),
                args.use_hinge_score,
                args.use_target_label,
                args.use_target_inputs,
                "".join(str(c) for c in args.cls_drop),
                fig_name,
            ),
        )

    torch.cuda.empty_cache()

    return


if __name__ == "__main__":
    args = argparser()
    dst_checkpoint_path = os.path.join(args.root_checkpoint_path, "best_val_loss.ckpt")

    # plot best trial
    plot_model(
        args,
        dst_checkpoint_path,
        "best",
        recompute_predictions=False,
        return_mean_logstd=args.return_mean_logstd,
    )