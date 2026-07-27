#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from glacier_mapping.data.data import load_band_names
from glacier_mapping.model.evaluation import (
    evaluate_full_split_modules,
    load_lightning_module,
    resolve_prediction_device,
)
from glacier_mapping.utils.gpu import cleanup_gpu_memory
from glacier_mapping.utils.config import load_servers_config


def clean_run_name(run_name: str) -> str:
    prefixes_to_remove = [
        "reproducibility_ci_",
        "reproducibility_dci_",
        "reproducibility_mc_",
        "ablation_ci_",
        "ablation_dci_",
        "ablation_mc_",
        "ablation2_ci_",
        "ablation2_dci_",
        "ablation2_mc_",
        "ci_",
        "dci_",
        "mc_",
    ]
    cleaned = run_name
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned


def find_best_checkpoint_for_run(run_name_pattern: str, server_config: dict) -> Path:
    server_output_path = Path(server_config["output_path"])
    best_checkpoint = None
    best_val_loss = float("inf")

    matching_dirs = []
    if server_output_path.exists():
        matching_dirs = [
            d
            for d in server_output_path.iterdir()
            if run_name_pattern in d.name and d.is_dir()
        ]

    for run_dir in matching_dirs:
        checkpoints_dir = run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue

        ckpts = list(checkpoints_dir.glob("*epoch=*.ckpt"))
        for ckpt in ckpts:
            if "val_loss=" in ckpt.name:
                try:
                    val_loss_str = ckpt.name.split("val_loss=")[1].split(".ckpt")[0]
                    val_loss = float(val_loss_str)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = ckpt
                except (ValueError, IndexError):
                    continue

    if best_checkpoint is None:
        raise ValueError(
            f"Could not find any checkpoints for run pattern '{run_name_pattern}' on server {server_config.get('hostname', 'unknown')}"
        )

    return best_checkpoint


def run_prediction_on_models(
    ci_checkpoint: Path | None,
    deb_checkpoint: Path | None,
    ci_threshold: float,
    deb_threshold: float,
    output_dir: Path,
    gpu_id: int,
    processed_data_path: str,
    split: str = "test",
    feature_importance: bool = False,
    fi_samples: int | None = None,
) -> dict:
    device = resolve_prediction_device(gpu_id)
    print(f"Prediction device: {device}")

    has_ci = ci_checkpoint is not None
    has_deb = deb_checkpoint is not None

    if not (has_ci or has_deb):
        raise ValueError("Must provide at least one checkpoint")

    frame_ci = None
    frame_deb = None

    if has_ci and ci_checkpoint is not None:
        frame_ci = load_lightning_module(ci_checkpoint, device, processed_data_path)

    if has_deb and deb_checkpoint is not None:
        frame_deb = load_lightning_module(deb_checkpoint, device, processed_data_path)

    result = evaluate_full_split_modules(
        frame_ci=frame_ci,
        frame_dci=frame_deb,
        split=split,
        ci_threshold=ci_threshold,
        dci_threshold=deb_threshold,
        output_dir=output_dir,
        save_probabilities=True,
    )

    if feature_importance and (frame_ci or frame_deb):
        saliency_frame = frame_ci if frame_ci else frame_deb
        if saliency_frame:
            print("Computing feature importance...")

            fi_output_dir = output_dir / "feature_importance"
            fi_output_dir.mkdir(parents=True, exist_ok=True)

            test_tiles = sorted(Path(saliency_frame.processed_dir, split).glob("tiff*"))
            importance_scores = compute_feature_importance(
                saliency_frame,
                test_tiles,
                target_class_idx=1,
                num_samples=fi_samples,
            )

            use_channels = saliency_frame.use_channels
            BAND_NAMES = load_band_names(saliency_frame.processed_dir)
            channel_names = BAND_NAMES[use_channels]

            save_feature_importance_results(
                importance_scores, channel_names, fi_output_dir
            )

    del frame_ci, frame_deb
    cleanup_gpu_memory(synchronize=False)

    return {
        "metrics": result.metrics,
        "output_dir": output_dir,
        "status": "SUCCESS",
    }


def main_prediction_logic(args) -> dict:
    try:
        servers_config = load_servers_config()

        if hasattr(args, "server") and args.server:
            current_server = args.server
            if current_server not in servers_config:
                raise ValueError(f"Server '{current_server}' not found in servers.yaml")
        else:
            current_server = None
            import socket

            hostname = socket.gethostname()

            for server_name, server_config in servers_config.items():
                if hostname == server_config.get("hostname", ""):
                    current_server = server_name
                    break

            if current_server is None:
                import os

                current_dir = os.getcwd()
                for server_name, server_config in servers_config.items():
                    if current_dir == server_config.get("code_path", ""):
                        current_server = server_name
                        break

            if current_server is None:
                raise RuntimeError(
                    "Could not determine current server. Check servers.yaml configuration or use --server parameter."
                )

        print(f"Using server: {current_server}")
        server_config = servers_config[current_server]

        ci_checkpoint_path = None
        deb_checkpoint_path = None

        if args.ci_run_name:
            ci_checkpoint_path = find_best_checkpoint_for_run(
                args.ci_run_name, server_config
            )
            print(f"Found CI checkpoint: {ci_checkpoint_path}")

        if args.deb_run_name:
            deb_checkpoint_path = find_best_checkpoint_for_run(
                args.deb_run_name, server_config
            )
            print(f"Found DCI checkpoint: {deb_checkpoint_path}")

        if hasattr(args, "output_dir") and args.output_dir:
            out_root = Path(args.output_dir)
        else:
            if args.ci_run_name and args.deb_run_name:
                base_run_name = clean_run_name(args.ci_run_name)
            elif args.ci_run_name:
                base_run_name = clean_run_name(args.ci_run_name)
            else:
                base_run_name = clean_run_name(args.deb_run_name)

            server_output_path = Path(server_config["output_path"])
            out_root = server_output_path.parent / "output_predictions" / base_run_name

        out_root.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {out_root}")

        processed_data_path = getattr(args, "processed_data_path", None)

        result = run_prediction_on_models(
            ci_checkpoint_path,
            deb_checkpoint_path,
            args.ci_threshold,
            args.deb_threshold,
            out_root,
            args.gpu,
            processed_data_path,
            args.split,
            getattr(args, "feature_importance", False),
            getattr(args, "fi_samples", None),
        )

        return result

    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def compute_feature_importance(
    module,
    test_tiles,
    target_class_idx,
    num_samples=None,
):
    module.eval()
    device = module.device
    use_channels = module.use_channels
    num_channels = len(use_channels)

    if num_samples is not None and num_samples < len(test_tiles):
        tiles_to_use = test_tiles[:num_samples]
    else:
        tiles_to_use = test_tiles

    print(f"Computing feature importance using {len(tiles_to_use)} test samples...")
    print(f"Target class index: {target_class_idx}")

    channel_gradients = np.zeros(num_channels, dtype=np.float64)

    for tile_path in tqdm(tiles_to_use, desc="Computing saliency"):
        x_full = np.load(tile_path)
        x = x_full[:, :, use_channels]

        x_norm = module.normalize(x)

        x_tensor = torch.from_numpy(x_norm).float().to(device).unsqueeze(0)
        x_tensor = x_tensor.permute(0, 3, 1, 2)
        x_tensor.requires_grad_(True)

        logits = module(x_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)

        target_prob = probs[0, target_class_idx, :, :].mean()

        target_prob.backward()

        channel_grad = np.zeros(len(use_channels))
        if x_tensor.grad is not None:
            grads = x_tensor.grad.data.abs()
            channel_grad = grads.sum(dim=(2, 3)).cpu().numpy()[0]

        channel_gradients += channel_grad

        x_tensor.grad = None
        del x_tensor, logits, probs

    channel_gradients /= len(tiles_to_use)

    return channel_gradients


def save_feature_importance_results(importance_scores, channel_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    total = importance_scores.sum()
    normalized = importance_scores / total if total > 0 else importance_scores

    df = pd.DataFrame(
        {
            "channel_idx": range(len(channel_names)),
            "channel_name": channel_names,
            "importance_score": importance_scores,
            "normalized_score": normalized,
        }
    )

    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    csv_path = output_dir / "channel_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance CSV: {csv_path}")

    print(f"\n{'=' * 70}")
    print("CHANNEL IMPORTANCE RANKING (ALL CHANNELS)")
    print(f"{'=' * 70}")
    print(
        f"{'Rank':<6}{'Idx':<6}{'Channel':<20}{'Raw Score':<15}{'Norm. Score':<15}{'Bar'}"
    )
    print("-" * 70)

    for rank, (i, row) in enumerate(df.iterrows()):
        bar_length = int(row["normalized_score"] * 50)
        bar = "█" * bar_length
        print(
            f"{rank + 1:<6}{row['channel_idx']:<6}{row['channel_name']:<20}"
            f"{row['importance_score']:<15.2f}{row['normalized_score']:<15.4f}{bar}"
        )


def get_checkpoint_paths(runs_dir, run_name, model_type):
    lightning_dir = runs_dir / run_name / "checkpoints"

    if lightning_dir.exists():
        if model_type == "all":
            ckpts = sorted(lightning_dir.glob("*.ckpt"))
            ckpt_pairs = []
            for ckpt in ckpts:
                if "epoch=" in ckpt.name:
                    epoch_part = ckpt.name.split("epoch=")[1].split("_")[0]
                    ckpt_name = f"epoch_{epoch_part}"
                elif "last.ckpt" in ckpt.name:
                    ckpt_name = "last"
                else:
                    ckpt_name = ckpt.stem
                ckpt_pairs.append((ckpt, ckpt_name))
            return ckpt_pairs
        else:
            if model_type == "best":
                ckpts = list(lightning_dir.glob("*epoch=*.ckpt"))
                if ckpts:
                    best_ckpt = max(
                        ckpts,
                        key=lambda x: int(x.name.split("epoch=")[1].split("_")[0]),
                    )
                    epoch_num = best_ckpt.name.split("epoch=")[1].split("_")[0]
                    return [(best_ckpt, f"epoch_{epoch_num}")]
                last_ckpt = lightning_dir / "last.ckpt"
                if last_ckpt.exists():
                    return [(last_ckpt, "last")]
            else:
                ckpt = lightning_dir / f"{model_type}.ckpt"
                if ckpt.exists():
                    return [(ckpt, model_type)]
                matching_ckpts = list(lightning_dir.glob(f"*{model_type}*.ckpt"))
                if matching_ckpts:
                    return [(matching_ckpts[0], model_type)]

            raise FileNotFoundError(
                f"Checkpoint not found: {lightning_dir / model_type}"
            )

    models_dir = runs_dir / run_name / "models"

    if model_type == "all":
        ckpts = sorted(models_dir.glob("model_*.pt"))
        ckpt_pairs = [(ckpt, ckpt.stem.replace("model_", "")) for ckpt in ckpts]
        return [(path, name) for path, name in ckpt_pairs if name != "best"]
    else:
        ckpt = models_dir / f"model_{model_type}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return [(ckpt, model_type)]


def load_existing_checkpoint_results(comparison_csv_path):
    if not comparison_csv_path.exists():
        return {}

    df = pd.read_csv(comparison_csv_path)
    results = {}

    for _, row in df.iterrows():
        ckpt_name = row["checkpoint"]
        metrics = row.to_dict()
        del metrics["checkpoint"]
        results[ckpt_name] = metrics

    return results


def identify_best_checkpoint(results_df, has_ci, has_deb, metric_strategy="IoU"):
    if has_ci and has_deb:
        if metric_strategy == "average_IoU":
            best_idx = (results_df["CI_IoU"] + results_df["Deb_IoU"]).idxmax()
            best_metric = (
                results_df.loc[best_idx, "CI_IoU"] + results_df.loc[best_idx, "Deb_IoU"]
            ) / 2
        elif metric_strategy == "CI_IoU":
            best_idx = results_df["CI_IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "CI_IoU"]
        elif metric_strategy == "Deb_IoU":
            best_idx = results_df["Deb_IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "Deb_IoU"]
        elif metric_strategy == "CI_Precision":
            best_idx = results_df["CI_P"].idxmax()
            best_metric = results_df.loc[best_idx, "CI_P"]
        elif metric_strategy == "Deb_Precision":
            best_idx = results_df["Deb_P"].idxmax()
            best_metric = results_df.loc[best_idx, "Deb_P"]
        else:
            best_idx = (results_df["CI_IoU"] + results_df["Deb_IoU"]).idxmax()
            best_metric = (
                results_df.loc[best_idx, "CI_IoU"] + results_df.loc[best_idx, "Deb_IoU"]
            ) / 2
    else:
        if metric_strategy == "IoU":
            best_idx = results_df["IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "IoU"]
        elif metric_strategy == "Precision":
            best_idx = results_df["P"].idxmax()
            best_metric = results_df.loc[best_idx, "P"]
        elif metric_strategy == "Recall":
            best_idx = results_df["R"].idxmax()
            best_metric = results_df.loc[best_idx, "R"]
        else:
            best_idx = results_df["IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "IoU"]

    best_checkpoint = results_df.loc[best_idx, "checkpoint"]
    return best_checkpoint, best_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run glacier mapping predictions")

    parser.add_argument(
        "--ci-run-name",
        type=str,
        help="CleanIce run name (will look in output/ directory)",
    )
    parser.add_argument(
        "--ci-threshold",
        type=float,
        default=0.5,
        help="CleanIce prediction threshold (default: 0.5)",
    )
    parser.add_argument(
        "--deb-run-name",
        type=str,
        help="Debris run name (will look in output/ directory)",
    )
    parser.add_argument(
        "--deb-threshold",
        type=float,
        default=0.5,
        help="Debris prediction threshold (default: 0.5)",
    )

    parser.add_argument(
        "--feature-importance",
        action="store_true",
        help="Compute feature importance (saliency)",
    )
    parser.add_argument(
        "--fi-samples",
        type=int,
        help="Number of samples for feature importance (default: all)",
    )
    parser.add_argument(
        "--fi-target-class",
        type=int,
        default=1,
        help="Target class for saliency (default: 1)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (default: 0). Use -1 for CPU.",
    )
    parser.add_argument(
        "--server",
        type=str,
        help="Server name from servers.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Explicit output directory for predictions (overrides default naming)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--processed-data-path",
        type=str,
        help="Override the dataset path saved in the checkpoint.",
    )

    args = parser.parse_args()

    result = main_prediction_logic(args)

    if result["status"] == "SUCCESS":
        print("\nPrediction complete.")
    else:
        print(f"\nPrediction failed: {result.get('error', 'Unknown error')}")
        exit(1)
