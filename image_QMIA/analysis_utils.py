import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from sklearn.metrics import roc_curve, auc

# Get base quantile performances


def get_rates(
    private_target_scores, public_target_scores, private_thresholds, public_thresholds
):
    # Get TPR, TNR and precision for all thresholds
    # scores are real valued vectors of size n
    # thresholds are either [n,n_thresholds] or [1,n_thresholds] depending on if the threshold is sample dependent or not
    assert (
        len(private_target_scores.shape) == 1
    ), "private scores need to be real-valued vectors"
    assert (
        len(public_target_scores.shape) == 1
    ), "public scores need to be real-valued vectors"
    assert (
        len(private_thresholds.shape) == 2
    ), "private thresholds need to be 2-d vectors"
    assert len(public_thresholds.shape) == 2, "public thresholds need to be 2-d vectors"
    prior = 0.0

    true_positives = (private_target_scores.reshape([-1, 1]) >= private_thresholds).sum(
        0
    ) + prior
    false_negatives = (private_target_scores.reshape([-1, 1]) < private_thresholds).sum(
        0
    ) + prior
    true_negatives = (public_target_scores.reshape([-1, 1]) < public_thresholds).sum(
        0
    ) + prior
    false_positives = (public_target_scores.reshape([-1, 1]) >= public_thresholds).sum(
        0
    ) + prior

    true_positive_rate = np.nan_to_num(
        true_positives / (true_positives + false_negatives)
    )
    true_negative_rate = np.nan_to_num(
        true_negatives / (true_negatives + false_positives)
    )
    precision = np.nan_to_num(
        true_positive_rate / (true_positive_rate + 1 - true_negative_rate)
    )

    return precision, true_positive_rate, true_negative_rate


def pinball_loss_np(target, score, quantile):
    target = target.reshape([-1, 1])
    assert (
        score.ndim == 2
    ), "score has the wrong shape, expected 2d input but got {}".format(score.shape)
    delta_score = target - score
    loss = np.maximum(delta_score * quantile, -delta_score * (1.0 - quantile)).mean(0)
    return loss


def pinball_loss_torch(
    target:    np.ndarray,    # shape [N]
    score:     np.ndarray,    # shape [1, Q] or [Q]
    quantile:  np.ndarray,    # shape [Q] or [1, Q]
) -> np.ndarray:              # returns [Q]
    t0 = time.time()
    device = torch.device("cuda")

    # [N,1]
    tgt = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(1)

    # score → tensor
    scr = torch.as_tensor(score, dtype=torch.float32, device=device)
    # if you passed in [Q], make it [1, Q]
    if scr.ndim == 1:
        scr = scr.unsqueeze(0)
    # if you passed in [1, Q], broadcast to [N, Q]
    if scr.shape[0] == 1:
        scr = scr.expand(tgt.shape[0], -1)
    # now scr is [N, Q]

    # quantile → tensor
    q = torch.as_tensor(quantile, dtype=torch.float32, device=device)
    # if you passed in [Q], make it [1, Q]
    if q.ndim == 1:
        q = q.unsqueeze(0)
    # now q is [1, Q], and will broadcast to [N, Q] automatically

    delta = tgt - scr               # [N, Q]
    loss  = torch.max(delta * q,    # [N, Q], since q broadcasts along dim0
                       -delta * (1.0 - q))
    loss  = loss.mean(dim=0)        # [Q]
    out   = loss.cpu().numpy()

    print(f"  ↳ pinball_loss_torch took {time.time() - t0:.3f}s")
    return out

# def plot_performance_curves(
#     private_target_scores,
#     public_target_scores,
#     private_predicted_score_thresholds=None,
#     public_predicted_score_thresholds=None,
#     model_target_quantiles=None,
#     model_name="Quantile Model",
#     use_quantile_thresholds=True,
#     use_thresholds=True,
#     use_logscale=True,
#     fontsize=12,
#     savefig_path="results.png",
#     plot_results=True,
# ):
#     plt.ioff()
#     n_baseline_points = 1000
#     if use_quantile_thresholds:
#         if use_logscale:
#             baseline_quantiles = np.sort(
#                 1.0 - np.logspace(-10, 0, n_baseline_points)[:-1]
#             )
#         else:
#             baseline_quantiles = np.linspace(0, 1, n_baseline_points)[:-1]
#         baseline_thresholds = np.quantile(public_target_scores, baseline_quantiles)
#         baseline_public_loss = pinball_loss_np(
#             public_target_scores,
#             baseline_thresholds.reshape([1, -1]),
#             baseline_quantiles,
#         )
#         baseline_private_loss = pinball_loss_np(
#             private_target_scores,
#             baseline_thresholds.reshape([1, -1]),
#             baseline_quantiles,
#         )

#     else:
#         raise NotImplementedError

#     baseline_precision, baseline_tpr, baseline_tnr = get_rates(
#         private_target_scores,
#         public_target_scores,
#         baseline_thresholds.reshape([1, -1]),
#         baseline_thresholds.reshape([1, -1]),
#     )

#     (
#         model_precision,
#         model_tpr,
#         model_tnr,
#         model_auc,
#         model_public_loss,
#         model_private_loss,
#     ) = (None, None, None, None, None, None)

#     if (
#         private_predicted_score_thresholds is not None and use_thresholds
#     ):  # scores and thresholds are provided directly (quantile model)
#         model_target_quantiles = np.sort(model_target_quantiles)
#         private_predicted_score_thresholds = np.sort(
#             private_predicted_score_thresholds, axis=-1
#         )
#         public_predicted_score_thresholds = np.sort(
#             public_predicted_score_thresholds, axis=-1
#         )

#         model_precision, model_tpr, model_tnr = get_rates(
#             private_target_scores,
#             public_target_scores,
#             private_predicted_score_thresholds,
#             public_predicted_score_thresholds,
#         )
#         model_public_loss = pinball_loss_np(
#             public_target_scores,
#             public_predicted_score_thresholds,
#             model_target_quantiles,
#         )
#         model_private_loss = pinball_loss_np(
#             private_target_scores,
#             private_predicted_score_thresholds,
#             model_target_quantiles,
#         )

#         model_adjusted_public_loss = pinball_loss_np(
#             public_target_scores, public_predicted_score_thresholds, model_tnr
#         )

#     # Plot ROC
#     fig, ax = plt.subplots(figsize=(6, 6), ncols=1, nrows=1)

#     ax.set_title("ROC", fontsize=fontsize)
#     ax.set_ylabel("True positive rate")
#     ax.set_xlabel("False positive rate")
#     ax.set_ylim([1e-3, 1])
#     ax.set_xlim([1e-3, 1])
#     baseline_auc = np.abs(np.trapz(baseline_tpr, x=1 - baseline_tnr))
#     # baseline_acc = (baseline_tpr + baseline_tnr).max() / 2.0
#     ax.plot(
#         1 - baseline_tnr,
#         baseline_tpr,
#         "-",
#         # label="Marginal Quantiles Acc {:.1f}%".format(100 * baseline_max_acc),
#         label="Marginal Quantiles",
#     )
#     if model_tpr is not None:
#         model_auc = np.abs(np.trapz(model_tpr, x=1 - model_tnr))
#         # model_acc = (model_tpr + model_tnr).max() / 2.0
#         ax.plot(
#             1 - model_tnr,
#             model_tpr,
#             "-",
#             markersize=12,
#             # label="{} Acc {:.1f}%".format(model_name, 100 * model_acc),
#             label="{}".format(model_name),
#         )

#     ax.legend()
#     if use_logscale:
#         plt.semilogx()
#         plt.semilogy()
#     # Finishing
#     plt.tight_layout()
#     if savefig_path is not None:
#         os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
#         roc_path = os.path.join(os.path.dirname(savefig_path), "roc.png")
#         plt.savefig(roc_path, dpi=300)
#         print("saving plot to", roc_path)
#     if plot_results:
#         plt.show()

#     # Plot Pinball losses on public data
#     fig, ax = plt.subplots(figsize=(6, 6), ncols=1, nrows=1)

#     ax.set_title("Pinball loss", fontsize=fontsize)
#     ax.set_xlabel("Significance level")
#     ax.set_ylabel("Pinball loss")
#     color = next(ax._get_lines.prop_cycler)["color"]
#     ax.plot(
#         1 - baseline_quantiles,
#         baseline_public_loss,
#         "x-",
#         label="Marginal Quantiles" + " (Public)",
#         color=color,
#     )
#     if model_public_loss is not None:
#         color = next(ax._get_lines.prop_cycler)["color"]
#         ax.plot(
#             1 - model_target_quantiles,
#             model_public_loss,
#             "x-",
#             label=model_name + "  (Public)",
#             color=color,
#         )
#     plt.semilogx()
#     ax.legend()
#     # Finishing
#     plt.tight_layout()
#     if savefig_path is not None:
#         os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
#         pinball_path = os.path.join(os.path.dirname(savefig_path), "pinball.png")
#         plt.savefig(pinball_path, dpi=300)
#         print("saving plot to", pinball_path)
#     if plot_results:
#         plt.show()

#     # pickle results and also print results at 1% and 0.1% FPR
#     pickle_path = os.path.join(
#         os.path.dirname(savefig_path),
#         os.path.basename(savefig_path).split(".")[0] + ".pkl",
#     )

#     def convenience_dict(
#         model_precision,
#         model_tpr,
#         model_tnr,
#         model_auc,
#         model_public_loss,
#         model_private_loss,
#         adjusted_public_loss=None,
#     ):
#         idx_1pc = np.argmin(np.abs(model_tnr - 0.99))
#         idx_01pc = np.argmin(np.abs(model_tnr - 0.999))
#         print(
#             "Precision @1%  FPR {:.2f}% \t  TPR @ 1% FPR {:.2f}% ".format(
#                 model_precision[idx_1pc] * 100, model_tpr[idx_1pc] * 100
#             )
#         )
#         print(
#             "Precision @0.1% FPR {:.2f}% \t  TPR @ 0.1% FPR {:.2f}% ".format(
#                 model_precision[idx_01pc] * 100, model_tpr[idx_01pc] * 100
#             )
#         )
#         cdict = {
#             "precision": model_precision,
#             "tpr": model_tpr,
#             "tnr": model_tnr,
#             "auc": model_auc,
#             "public_loss": model_public_loss,
#             "private_loss": model_private_loss,
#         }
#         cdict["adjusted_public_loss"] = (
#             adjusted_public_loss
#             if adjusted_public_loss is not None
#             else model_public_loss
#         )
#         return cdict

#     with open(pickle_path, "wb") as f:
#         save_dict = {}
#         if baseline_tnr is not None:
#             print("baseline")
#             save_dict["baseline"] = convenience_dict(
#                 baseline_precision,
#                 baseline_tpr,
#                 baseline_tnr,
#                 baseline_auc,
#                 baseline_public_loss,
#                 baseline_private_loss,
#             )

#         if model_tpr is not None:
#             print("model")
#             save_dict["model"] = convenience_dict(
#                 model_precision,
#                 model_tpr,
#                 model_tnr,
#                 model_auc,
#                 model_public_loss,
#                 model_private_loss,
#                 model_adjusted_public_loss,
#             )
#         pickle.dump(save_dict, f)

#     return baseline_auc, model_auc

def plot_performance_curves(
    private_target_scores: np.ndarray,
    public_target_scores:  np.ndarray,
    private_predicted_score_thresholds: np.ndarray = None,
    public_predicted_score_thresholds:  np.ndarray = None,
    private_p_values:       np.ndarray = None,
    public_p_values:        np.ndarray = None,
    model_target_quantiles: np.ndarray = None,
    model_name:            str       = "Quantile Model",
    use_quantile_thresholds: bool     = True,
    use_thresholds:          bool     = True,
    use_logscale:            bool     = True,
    fontsize:               int       = 12,
    savefig_path:           str       = "results.png",
    plot_results:           bool      = True,
):
    print("[plot_performance_curves] Starting...")
    start_time = time.time()
    plt.ioff()

    # ---- 1) Baseline marginal-quantile curves ----
    print("[plot_performance_curves] Computing baseline curves...")
    n_baseline = 1000
    if not use_quantile_thresholds:
        raise NotImplementedError("Only quantile baselines supported.")
    baseline_q = (
        np.sort(1.0 - np.logspace(-10, 0, n_baseline)[:-1])
        if use_logscale else np.linspace(0, 1, n_baseline)[:-1]
    )
    print(f"  Baseline uses {len(baseline_q)} quantiles, logscale={use_logscale}")

    baseline_thr = np.quantile(public_target_scores, baseline_q)
    print("  Calculated baseline thresholds.")
    baseline_pub_loss = pinball_loss_torch(
        public_target_scores, baseline_thr.reshape(1, -1), baseline_q)
    baseline_priv_loss = pinball_loss_torch(
        private_target_scores, baseline_thr.reshape(1, -1), baseline_q)
    print("  Computed baseline pinball losses.")
    print(f"  Baseline step done in {time.time()-start_time:.2f}s")

    baseline_prec, baseline_tpr, baseline_tnr = get_rates(
        private_target_scores,
        public_target_scores,
        baseline_thr.reshape(1, -1),
        baseline_thr.reshape(1, -1),
    )
    baseline_auc = np.abs(np.trapz(baseline_tpr, x=1 - baseline_tnr))
    print(f"  Baseline AUC: {baseline_auc:.4f}")

    # ---- 2) Model ROC: p-value or thresholds ----
    print("[plot_performance_curves] Building model ROC...")
    model_prec = model_tpr = model_tnr = model_auc = None
    model_pub_loss = model_priv_loss = None

    if private_p_values is not None and public_p_values is not None:
        print("  Using p-values branch (dense ROC)")
        y_true  = np.concatenate([np.ones_like(private_p_values),
                                  np.zeros_like(public_p_values)])
        y_score = np.concatenate([private_p_values, public_p_values])
        fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
        print(f"  ROC curve generated with {len(thr)} thresholds")

        model_tpr = tpr
        model_tnr = 1 - fpr
        model_auc = np.abs(np.trapz(model_tpr, x=1 - model_tnr))
        P = len(private_p_values)
        N = len(public_p_values)
        model_prec = (tpr * P) / (tpr * P + fpr * N)
        model_target_quantiles = thr[np.newaxis, :]
        print(f"  Model AUC (p-values): {model_auc:.4f}")

    elif (private_predicted_score_thresholds is not None
          and public_predicted_score_thresholds is not None
          and use_thresholds):
        print("  Using thresholds branch")
        model_target_quantiles = np.sort(model_target_quantiles)
        private_predicted_score_thresholds = np.sort(
            private_predicted_score_thresholds, axis=-1)
        public_predicted_score_thresholds  = np.sort(
            public_predicted_score_thresholds,  axis=-1)

        model_prec, model_tpr, model_tnr = get_rates(
            private_target_scores,
            public_target_scores,
            private_predicted_score_thresholds,
            public_predicted_score_thresholds,
        )
        model_pub_loss = pinball_loss_np(
            public_target_scores,
            public_predicted_score_thresholds,
            model_target_quantiles,
        )
        model_priv_loss = pinball_loss_np(
            private_target_scores,
            private_predicted_score_thresholds,
            model_target_quantiles,
        )
        model_auc = np.abs(np.trapz(model_tpr, x=1 - model_tnr))
        print(f"  Model AUC (thresholds): {model_auc:.4f}")

    else:
        print("  No model predictions provided; skipping model ROC.")
    print(f"  Model ROC step done in {time.time()-start_time:.2f}s")

    # ---- 3) Plot ROC ----
    print("[plot_performance_curves] Plotting ROC...")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title("ROC", fontsize=fontsize)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-3, 1)

    ax.plot(1 - baseline_tnr, baseline_tpr,
            "-", label=f"Marginal Quantiles (AUC={baseline_auc:.3f})")
    if model_tpr is not None:
        ax.plot(1 - model_tnr, model_tpr,
                "-", label=f"{model_name} (AUC={model_auc:.3f})")
    ax.legend()
    if use_logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    plt.tight_layout()

    if savefig_path:
        roc_path = os.path.join(os.path.dirname(savefig_path), "roc.png")
        print(f"  Saving ROC plot to {roc_path}")
        os.makedirs(os.path.dirname(roc_path), exist_ok=True)
        plt.savefig(roc_path, dpi=300)
    if plot_results:
        plt.show()
    print(f"  ROC plot done in {time.time()-start_time:.2f}s")

    # ---- 4) Pinball loss (public) ----
    print("[plot_performance_curves] Plotting Pinball loss...")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title("Pinball loss", fontsize=fontsize)
    ax.set_xlabel("Significance level")
    ax.set_ylabel("Pinball loss")

    ax.plot(1 - baseline_q, baseline_pub_loss, "x-",
            label="Marginal Quantiles (Public)")
    if model_pub_loss is not None:
        ax.plot(1 - model_target_quantiles.flatten(), model_pub_loss, "x-",
                label=f"{model_name} (Public)")
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()

    if savefig_path:
        pinball_path = os.path.join(os.path.dirname(savefig_path), "pinball.png")
        print(f"  Saving Pinball plot to {pinball_path}")
        os.makedirs(os.path.dirname(pinball_path), exist_ok=True)
        plt.savefig(pinball_path, dpi=300)
    if plot_results:
        plt.show()
    print(f"  Pinball plot done in {time.time()-start_time:.2f}s")

    # ---- 5) Save results ----
    print("[plot_performance_curves] Saving numeric results...")
    def make_dict(prec, tpr, tnr, auc, pub_loss, priv_loss):
        idx1  = np.argmin(np.abs(tnr - 0.99))
        idx01 = np.argmin(np.abs(tnr - 0.999))
        print(f"  Precision @1% FPR: {prec[idx1]*100:.2f}%  TPR: {tpr[idx1]*100:.2f}%")
        print(f"  Precision @0.1% FPR: {prec[idx01]*100:.2f}%  TPR: {tpr[idx01]*100:.2f}%")
        return {"precision": prec, "tpr": tpr, "tnr": tnr,
                "auc": auc, "public_loss": pub_loss, "private_loss": priv_loss}
    
    results = {"baseline": make_dict(
                   baseline_prec, baseline_tpr, baseline_tnr,
                   baseline_auc, baseline_pub_loss, baseline_priv_loss)}

    if model_tpr is not None:
        results["model"] = make_dict(
            model_prec, model_tpr, model_tnr,
            model_auc, model_pub_loss, model_priv_loss
        )

    # if savefig_path:
    #     pkl = os.path.splitext(savefig_path)[0] + ".pkl"
    #     with open(pkl, "wb") as f:
    #         pickle.dump(results, f)
    #     print("saved metrics to", pkl)

    return baseline_auc, model_auc


# def plot_performance_curves_id_ood(
#     private_target_scores,
#     public_target_scores,
#     private_predicted_score_thresholds=None,
#     public_predicted_score_thresholds=None,
#     model_target_quantiles=None,
#     model_name="Quantile Model",
#     private_id_mask=None,  # Boolean mask for ID data in private set
#     private_ood_mask=None,  # Boolean mask for OOD data in private set
#     public_id_mask=None,    # Boolean mask for ID data in public set
#     public_ood_mask=None,   # Boolean mask for OOD data in public set 
#     use_quantile_thresholds=True,
#     use_thresholds=True,
#     use_logscale=True,
#     fontsize=12,
#     savefig_path="results.png",
#     plot_results=True,
# ):
#     """
#     Plot ID and OOD performance curves side by side.
#     """
#     plt.ioff()
    
#     # Create a figure with 2 rows and 2 columns
#     fig, axes = plt.subplots(figsize=(12, 10), nrows=2, ncols=2)
    
#     # Define dataset types and their corresponding masks
#     dataset_types = ["ID", "OOD"]
#     private_masks = [private_id_mask, private_ood_mask]
#     public_masks = [public_id_mask, public_ood_mask]
    
#     # If no masks provided, default to all data as ID
#     if private_id_mask is None and private_ood_mask is None:
#         private_masks = [
#             np.ones_like(private_target_scores, dtype=bool),  # All data as ID
#             np.zeros_like(private_target_scores, dtype=bool)  # No OOD data
#         ]
    
#     if public_id_mask is None and public_ood_mask is None:
#         public_masks = [
#             np.ones_like(public_target_scores, dtype=bool),  # All data as ID
#             np.zeros_like(public_target_scores, dtype=bool)  # No OOD data
#         ]
    
#     # Define convenience_dict function similar to the first function
#     def convenience_dict(
#         model_precision,
#         model_tpr,
#         model_tnr,
#         model_auc,
#         model_public_loss,
#         model_private_loss,
#         adjusted_public_loss=None,
#     ):
#         idx_1pc = np.argmin(np.abs(model_tnr - 0.99))
#         idx_01pc = np.argmin(np.abs(model_tnr - 0.999))
#         print(
#             "Precision @1%  FPR {:.2f}% \t  TPR @ 1% FPR {:.2f}% ".format(
#                 model_precision[idx_1pc] * 100, model_tpr[idx_1pc] * 100
#             )
#         )
#         print(
#             "Precision @0.1% FPR {:.2f}% \t  TPR @ 0.1% FPR {:.2f}% ".format(
#                 model_precision[idx_01pc] * 100, model_tpr[idx_01pc] * 100
#             )
#         )
#         cdict = {
#             "precision": model_precision,
#             "tpr": model_tpr,
#             "tnr": model_tnr,
#             "auc": model_auc,
#             "public_loss": model_public_loss,
#             "private_loss": model_private_loss,
#         }
#         cdict["adjusted_public_loss"] = (
#             adjusted_public_loss
#             if adjusted_public_loss is not None
#             else model_public_loss
#         )
#         return cdict
    
#     # For each dataset type (ID and OOD)
#     results = {}
#     for i, (dataset_type, priv_mask, pub_mask) in enumerate(zip(dataset_types, private_masks, public_masks)):
#         # Skip if no data for this type in either dataset
#         if not torch.any(priv_mask) or not torch.any(pub_mask):
#             continue
            
#         # Filter data based on masks
#         filtered_private_scores = private_target_scores[priv_mask]
#         filtered_public_scores = public_target_scores[pub_mask]
        
#         # Calculate baseline metrics
#         n_baseline_points = 4000
#         if use_quantile_thresholds:
#             if use_logscale:
#                 baseline_quantiles = np.sort(
#                     1.0 - np.logspace(-16, 0, n_baseline_points)[:-1]
#                 )
#             else:
#                 baseline_quantiles = np.linspace(0, 1, n_baseline_points)[:-1]
#             baseline_thresholds = np.quantile(filtered_public_scores, baseline_quantiles)
#             baseline_public_loss = pinball_loss_np(
#                 filtered_public_scores,
#                 baseline_thresholds.reshape([1, -1]),
#                 baseline_quantiles,
#             )
#             baseline_private_loss = pinball_loss_np(
#                 filtered_private_scores,
#                 baseline_thresholds.reshape([1, -1]),
#                 baseline_quantiles,
#             )
#         else:
#             raise NotImplementedError

#         baseline_precision, baseline_tpr, baseline_tnr = get_rates(
#             filtered_private_scores,
#             filtered_public_scores,
#             baseline_thresholds.reshape([1, -1]),
#             baseline_thresholds.reshape([1, -1]),
#         )
        
#         # Calculate baseline AUC
#         baseline_auc = np.abs(np.trapz(baseline_tpr, x=1 - baseline_tnr))
        
#         # Calculate model metrics if available
#         model_precision, model_tpr, model_tnr = None, None, None
#         model_public_loss, model_private_loss = None, None
#         model_adjusted_public_loss = None
#         model_auc = None
        
#         if (private_predicted_score_thresholds is not None and use_thresholds):
#             filtered_private_thresholds = private_predicted_score_thresholds[priv_mask]
#             filtered_public_thresholds = public_predicted_score_thresholds[pub_mask]
            
#             model_target_quantiles = np.sort(model_target_quantiles)
#             filtered_private_thresholds = np.sort(filtered_private_thresholds, axis=-1)
#             filtered_public_thresholds = np.sort(filtered_public_thresholds, axis=-1)

#             model_precision, model_tpr, model_tnr = get_rates(
#                 filtered_private_scores,
#                 filtered_public_scores,
#                 filtered_private_thresholds,
#                 filtered_public_thresholds,
#             )
#             model_public_loss = pinball_loss_np(
#                 filtered_public_scores,
#                 filtered_public_thresholds,
#                 model_target_quantiles,
#             )
#             model_private_loss = pinball_loss_np(
#                 filtered_private_scores,
#                 filtered_private_thresholds,
#                 model_target_quantiles,
#             )
            
#             model_adjusted_public_loss = pinball_loss_np(
#                 filtered_public_scores, 
#                 filtered_public_thresholds, 
#                 model_tnr
#             )
            
#             model_auc = np.abs(np.trapz(model_tpr, x=1 - model_tnr))
        
#         # Plot ROC curve
#         ax_roc = axes[0, i]
#         ax_roc.set_title(f"{dataset_type} ROC", fontsize=fontsize)
#         ax_roc.set_ylabel("True positive rate")
#         ax_roc.set_xlabel("False positive rate")
#         ax_roc.set_ylim([1e-3, 1])
#         ax_roc.set_xlim([1e-3, 1])
        
#         ax_roc.plot(
#             1 - baseline_tnr,
#             baseline_tpr,
#             "-",
#             label=f"Marginal Quantiles (AUC={baseline_auc:.3f})",
#         )
        
#         if model_tpr is not None:
#             ax_roc.plot(
#                 1 - model_tnr,
#                 model_tpr,
#                 "-",
#                 markersize=12,
#                 label=f"{model_name} (AUC={model_auc:.3f})",
#             )

#         ax_roc.legend()
#         if use_logscale:
#             ax_roc.set_xscale('log')
#             ax_roc.set_yscale('log')
        
#         # Plot Pinball loss
#         ax_pinball = axes[1, i]
#         ax_pinball.set_title(f"{dataset_type} Pinball loss", fontsize=fontsize)
#         ax_pinball.set_xlabel("Significance level")
#         ax_pinball.set_ylabel("Pinball loss")
        
#         ax_pinball.plot(
#             1 - baseline_quantiles,
#             baseline_public_loss,
#             "x-",
#             label="Marginal Quantiles (Public)",
#         )
        
#         if model_public_loss is not None:
#             ax_pinball.plot(
#                 1 - model_target_quantiles,
#                 model_public_loss,
#                 "x-",
#                 label=f"{model_name} (Public)",
#             )
        
#         ax_pinball.set_xscale('log')
#         ax_pinball.legend()
        
#         # Print and store results using convenience_dict
#         print(f"\n{dataset_type} dataset:")
        
#         print("baseline")
#         results[dataset_type] = {"baseline": convenience_dict(
#             baseline_precision,
#             baseline_tpr,
#             baseline_tnr,
#             baseline_auc,
#             baseline_public_loss,
#             baseline_private_loss,
#         )}
        
#         if model_tpr is not None:
#             print("model")
#             results[dataset_type]["model"] = convenience_dict(
#                 model_precision,
#                 model_tpr,
#                 model_tnr,
#                 model_auc,
#                 model_public_loss,
#                 model_private_loss,
#                 model_adjusted_public_loss,
#             )
    
#     # Adjust layout and save
#     plt.tight_layout()
#     if savefig_path is not None:
#         os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
#         plt.savefig(savefig_path, dpi=300)
#         print(f"Saving plot to {savefig_path}")
        
#         # Also save pickle with results
#         pickle_path = os.path.join(
#             os.path.dirname(savefig_path),
#             os.path.basename(savefig_path).split(".")[0] + ".pkl",
#         )
        
#         with open(pickle_path, "wb") as f:
#             pickle.dump(results, f)
#             print(f"Saving results to {pickle_path}")
            
#     if plot_results:
#         plt.show()
    
#     # Return baseline and model AUCs from ID dataset (if available)
#     id_baseline_auc = results.get("ID", {}).get("baseline", {}).get("auc", None) if "ID" in results else None
#     id_model_auc = results.get("ID", {}).get("model", {}).get("auc", None) if "ID" in results and "model" in results["ID"] else None
    
#     return id_baseline_auc, id_model_auc

# def plot_performance_curves_id_ood_with_probs(
#     private_target_scores,
#     public_target_scores,
#     private_target_probs=None,
#     public_target_probs=None,
#     model_name="Gaussian Model",
#     private_id_mask=None,  # Boolean mask for ID data in private set
#     private_ood_mask=None,  # Boolean mask for OOD data in private set
#     public_id_mask=None,    # Boolean mask for ID data in public set
#     public_ood_mask=None,   # Boolean mask for OOD data in public set 
#     use_logscale=True,
#     fontsize=12,
#     savefig_path="results_prob.png",
#     plot_results=True,
# ):
#     """
#     Plot ID and OOD performance curves side by side using probability density values.
#     """
    
#     plt.ioff()
    
#     # Create a figure with 2 rows and 2 columns
#     fig, axes = plt.subplots(figsize=(12, 10), nrows=2, ncols=2)
    
#     # Define dataset types and their corresponding masks
#     dataset_types = ["ID", "OOD"]
#     private_masks = [private_id_mask, private_ood_mask]
#     public_masks = [public_id_mask, public_ood_mask]
    
#     # If no masks provided, default to all data as ID
#     if private_id_mask is None and private_ood_mask is None:
#         private_masks = [
#             np.ones_like(private_target_scores, dtype=bool),  # All data as ID
#             np.zeros_like(private_target_scores, dtype=bool)  # No OOD data
#         ]
    
#     if public_id_mask is None and public_ood_mask is None:
#         public_masks = [
#             np.ones_like(public_target_scores, dtype=bool),  # All data as ID
#             np.zeros_like(public_target_scores, dtype=bool)  # No OOD data
#         ]
    
#     # Define convenience_dict function
#     def convenience_dict(
#         model_precision,
#         model_tpr,
#         model_tnr,
#         model_auc,
#     ):
#         idx_1pc = np.argmin(np.abs(model_tnr - 0.99))
#         idx_01pc = np.argmin(np.abs(model_tnr - 0.999))
#         print(
#             "Precision @1%  FPR {:.2f}% \t  TPR @ 1% FPR {:.2f}% ".format(
#                 model_precision[idx_1pc] * 100, model_tpr[idx_1pc] * 100
#             )
#         )
#         print(
#             "Precision @0.1% FPR {:.2f}% \t  TPR @ 0.1% FPR {:.2f}% ".format(
#                 model_precision[idx_01pc] * 100, model_tpr[idx_01pc] * 100
#             )
#         )
#         return {
#             "precision": model_precision,
#             "tpr": model_tpr,
#             "tnr": model_tnr,
#             "auc": model_auc,
#         }
    
#     # For each dataset type (ID and OOD)
#     results = {}
#     for i, (dataset_type, priv_mask, pub_mask) in enumerate(zip(dataset_types, private_masks, public_masks)):
#         # Skip if no data for this type in either dataset
#         if not torch.any(priv_mask) or not torch.any(pub_mask):
#             continue
            
#         # Filter data based on masks
#         filtered_private_scores = private_target_scores[priv_mask]
#         filtered_public_scores = public_target_scores[pub_mask]
        
#         # Calculate baseline metrics using quantiles (similar to original function)
#         n_baseline_points = 500
#         if use_logscale:
#             baseline_quantiles = np.sort(
#                 1.0 - np.logspace(-6, 0, n_baseline_points)[:-1]
#             )
#         else:
#             baseline_quantiles = np.linspace(0, 1, n_baseline_points)[:-1]
#         baseline_thresholds = np.quantile(filtered_public_scores, baseline_quantiles)
        
#         # Create binary labels for ROC calculation
#         # 1 for private (positive class), 0 for public (negative class)
#         y_true = np.hstack([
#             np.ones(len(filtered_private_scores)),
#             np.zeros(len(filtered_public_scores))
#         ])
        
#         # Get baseline scores
#         y_scores_baseline = np.hstack([
#             filtered_private_scores,
#             filtered_public_scores
#         ])
        
#         # Calculate baseline ROC using scikit-learn
#         baseline_fpr, baseline_tpr, baseline_thresholds_roc = roc_curve(y_true, y_scores_baseline)
#         baseline_auc = auc(baseline_fpr, baseline_tpr)
        
#         # Calculate baseline precision
#         baseline_precision = []
#         for t in baseline_thresholds_roc:
#             tp = np.sum((y_scores_baseline >= t) & (y_true == 1))
#             fp = np.sum((y_scores_baseline >= t) & (y_true == 0))
#             baseline_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
#         baseline_precision = np.array(baseline_precision)
#         baseline_tnr = 1 - baseline_fpr
        
#         # Calculate model metrics using probability values if available
#         model_fpr, model_tpr, model_thresholds_roc = None, None, None
#         model_precision, model_auc = None, None
        
#         if private_target_probs is not None and public_target_probs is not None:
#             # Filter probability values based on masks
#             filtered_private_probs = private_target_probs[priv_mask]
#             filtered_public_probs = public_target_probs[pub_mask]
            
#             # Stack probabilities for scikit-learn
#             y_scores_model = np.hstack([
#                 filtered_private_probs,
#                 filtered_public_probs
#             ])
            
#             # Calculate ROC using scikit-learn
#             model_fpr, model_tpr, model_thresholds_roc = roc_curve(y_true, y_scores_model)
#             model_auc = auc(model_fpr, model_tpr)
            
#             # Calculate precision
#             model_precision = []
#             for t in model_thresholds_roc:
#                 tp = np.sum((y_scores_model >= t) & (y_true == 1))
#                 fp = np.sum((y_scores_model >= t) & (y_true == 0))
#                 model_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
#             model_precision = np.array(model_precision)
#             model_tnr = 1 - model_fpr
        
#         # Plot ROC curve
#         ax_roc = axes[0, i]
#         ax_roc.set_title(f"{dataset_type} ROC", fontsize=fontsize)
#         ax_roc.set_ylabel("True positive rate")
#         ax_roc.set_xlabel("False positive rate")
#         ax_roc.set_ylim([1e-3, 1])
#         ax_roc.set_xlim([1e-3, 1])
        
#         ax_roc.plot(
#             baseline_fpr,
#             baseline_tpr,
#             "-",
#             label=f"Marginal Quantiles (AUC={baseline_auc:.3f})",
#         )
        
#         if model_tpr is not None:
#             ax_roc.plot(
#                 model_fpr,
#                 model_tpr,
#                 "-",
#                 markersize=12,
#                 label=f"{model_name} (AUC={model_auc:.3f})",
#             )

#         ax_roc.legend()
#         if use_logscale:
#             ax_roc.set_xscale('log')
#             ax_roc.set_yscale('log')
        
#         # Plot probability distribution histograms for the second row
#         ax_prob = axes[1, i]
#         ax_prob.set_title(f"{dataset_type} Probability Distribution", fontsize=fontsize)
#         ax_prob.set_xlabel("Log Probability Density")
#         ax_prob.set_ylabel("Count")
        
#         if private_target_probs is not None and public_target_probs is not None:
#             # Plot histograms of the log probability values
#             bins = np.linspace(
#                 min(np.log10(filtered_private_probs.clip(min=1e-10)).min(), 
#                     np.log10(filtered_public_probs.clip(min=1e-10)).min()), 
#                 max(np.log10(filtered_private_probs.clip(min=1e-10)).max(), 
#                     np.log10(filtered_public_probs.clip(min=1e-10)).max()), 
#                 50
#             )
            
#             ax_prob.hist(
#                 np.log10(filtered_private_probs.clip(min=1e-10)), 
#                 bins=bins, 
#                 alpha=0.6, 
#                 label="Private Set"
#             )
#             ax_prob.hist(
#                 np.log10(filtered_public_probs.clip(min=1e-10)), 
#                 bins=bins, 
#                 alpha=0.6, 
#                 label="Public Set"
#             )
#             ax_prob.legend()
        
#         # Print and store results
#         print(f"\n{dataset_type} dataset:")
        
#         print("baseline")
#         results[dataset_type] = {"baseline": convenience_dict(
#             baseline_precision,
#             baseline_tpr,
#             baseline_tnr,
#             baseline_auc,
#         )}
        
#         if model_tpr is not None:
#             print("model")
#             results[dataset_type]["model"] = convenience_dict(
#                 model_precision,
#                 model_tpr,
#                 model_tnr,
#                 model_auc,
#             )
    
#     # Adjust layout and save
#     plt.tight_layout()
#     if savefig_path is not None:
#         os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
#         plt.savefig(savefig_path, dpi=300)
#         print(f"Saving plot to {savefig_path}")
        
#         # Also save pickle with results
#         pickle_path = os.path.join(
#             os.path.dirname(savefig_path),
#             os.path.basename(savefig_path).split(".")[0] + ".pkl",
#         )
        
#         with open(pickle_path, "wb") as f:
#             pickle.dump(results, f)
#             print(f"Saving results to {pickle_path}")
            
#     if plot_results:
#         plt.show()
    
#     # Return baseline and model AUCs from ID dataset (if available)
#     id_baseline_auc = results.get("ID", {}).get("baseline", {}).get("auc", None) if "ID" in results else None
#     id_model_auc = results.get("ID", {}).get("model", {}).get("auc", None) if "ID" in results and "model" in results["ID"] else None
    
#     return id_baseline_auc, id_model_auc

def plot_performance_curves_id_ood(
    private_target_scores: np.ndarray,
    public_target_scores:  np.ndarray,
    private_predicted_score_thresholds: np.ndarray = None,
    public_predicted_score_thresholds:  np.ndarray = None,
    model_target_quantiles=None,
    private_p_values:       np.ndarray = None,
    public_p_values:        np.ndarray = None,
    model_name:            str       = "Quantile Model",
    private_id_mask:       np.ndarray = None,
    private_ood_mask:      np.ndarray = None,
    public_id_mask:        np.ndarray = None,
    public_ood_mask:       np.ndarray = None,
    use_logscale:            bool     = True,
    fontsize:               int       = 12,
    savefig_path:           str       = "results_id_ood.png",
    plot_results:           bool      = True,
):
    plt.ioff()

    # default masks
    Npriv, Npub = len(private_target_scores), len(public_target_scores)
    if private_id_mask is None:
        private_id_mask  = np.ones(Npriv, dtype=bool)
        private_ood_mask = np.zeros(Npriv, dtype=bool)
    if public_id_mask is None:
        public_id_mask  = np.ones(Npub, dtype=bool)
        public_ood_mask = np.zeros(Npub, dtype=bool)

    # make two rows, one column
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cols = [("OOD", private_ood_mask, public_ood_mask),
            ("ID",  private_id_mask,  public_id_mask)]
    
    for ax, (label, priv_mask, pub_mask) in zip(axes, cols):
        # slice out only ID or OOD
        ps  = private_target_scores[priv_mask]
        qs  = public_target_scores[ pub_mask]
        ppv = None if private_p_values is None else private_p_values[priv_mask]
        pqv = None if public_p_values  is None else public_p_values[ pub_mask]

        # --- 1) marginal‐quantile ROC ---
        # quantile levels for baseline ROC
        q_levels = np.sort(1.0 - np.logspace(-10, 0, 1000)[:-1]) if use_logscale \
                   else np.linspace(0,1,1000)[:-1]
        thr = np.quantile(qs, q_levels)             # [Q]
        # get_rates should return (prec, tpr, tnr) arrays of length Q
        prec_b, tpr_b, tnr_b = get_rates(ps, qs,
                                         thr[np.newaxis, :],
                                         thr[np.newaxis, :])
        fpr_b = 1 - tnr_b

        # --- 2) model‐based ROC ---
        fpr_m = tpr_m = None
        if ppv is not None and pqv is not None:
            # p‐value branch
            y_true  = np.concatenate([np.ones_like(ppv),
                                      np.zeros_like(pqv)])
            y_score = np.concatenate([ppv, pqv])
            fpr_m, tpr_m, _ = roc_curve(y_true, y_score,
                                        drop_intermediate=False)
        elif (private_predicted_score_thresholds is not None
              and public_predicted_score_thresholds is not None):
            # thresholds branch
            thr_priv = np.sort(private_predicted_score_thresholds[priv_mask],
                               axis=-1)
            thr_pub  = np.sort(public_predicted_score_thresholds[pub_mask],
                               axis=-1)
            _, tpr_m, tnr_m = get_rates(ps, qs, thr_priv, thr_pub)
            fpr_m = 1 - tnr_m

        # --- Plot ROC ---
        ax.set_title(f"{label} ROC", fontsize=fontsize)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(1e-3, 1)

        ax.plot(fpr_b, tpr_b, "-", label="Marginal Quantiles")
        if fpr_m is not None:
            ax.plot(fpr_m, tpr_m, "-", label=model_name)

        if use_logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        # --- Annotate TPR at 1% and 0.1% FPR ---
        for thresh, pct in [(0.01, "1%"), (0.001, "0.1%")]:
            # baseline
            idx_b = np.argmin(np.abs(fpr_b - thresh))
            tpr_b_at = tpr_b[idx_b]
            if tpr_b_at == 0:
                tpr_b_at = np.interp(thresh, fpr_b, tpr_b)
            ax.text(0.02, 0.9 - 0.05*(0 if pct=="1%" else 1),
                    f"Baseline TPR @ {pct} FPR: {tpr_b_at*100:.1f}%",
                    transform=ax.transAxes,
                    fontsize=fontsize-2,
                    color="C0")

            # model, if present
            if fpr_m is not None:
                idx_m = np.argmin(np.abs(fpr_m - thresh))
                tpr_m_at = tpr_m[idx_m]
                if tpr_m_at == 0:
                    tpr_m_at = np.interp(thresh, fpr_m, tpr_m)
                ax.text(0.02, 0.8 - 0.05*(0 if pct=="1%" else 1),
                        f"{model_name} TPR @ {pct} FPR: {tpr_m_at*100:.1f}%",
                        transform=ax.transAxes,
                        fontsize=fontsize-2,
                        color="C1")

        ax.legend(fontsize=fontsize-2)

    plt.tight_layout()

    # save the combined figure
    if savefig_path:
        d = os.path.dirname(savefig_path)
        if d:
            os.makedirs(d, exist_ok=True)
        print(f"[plot_id_ood] Saving combined figure to {savefig_path}")
        fig.savefig(savefig_path, dpi=300)

    if plot_results:
        plt.show()

    # return just the ID‐case AUCs
    auc_b = np.abs(np.trapz(tpr_b, x=fpr_b))
    auc_m = (np.abs(np.trapz(tpr_m, x=fpr_m))
             if fpr_m is not None else None)
    return auc_b, auc_m
