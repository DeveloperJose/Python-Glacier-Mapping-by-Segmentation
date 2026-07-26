from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from glacier_mapping.utils.gpu import cleanup_gpu_memory
from glacier_mapping.utils.callback_utils import (
    load_dataset_metadata,
    generate_single_visualization,
    select_slices_by_iou_thirds,
    parse_slice_path,
    select_informative_test_tiles,
    log_visualizations_to_all_loggers,
)
import glacier_mapping.utils.logging as log


class ValidationVisualizationCallback(Callback):
    def __init__(
        self,
        viz_n: int = 4,
        log_every_n_epochs: int = 10,
        selection: str = "iou",
        save_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        scale_factor: float = 1,
    ):
        super().__init__()
        self.viz_n = viz_n
        self.log_every_n_epochs = log_every_n_epochs
        self.selection = selection
        self.save_dir = Path(save_dir) if save_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.scale_factor = scale_factor
        self.selected_slice_paths: Optional[List[Path]] = None
        self.slice_metadata: Dict[Path, Tuple[int, int]] = {}
        self.tile_rank_map: Dict[Path, int] = {}
        self._metadata_cache = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if self.log_every_n_epochs == 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            self._generate_visualizations(trainer, pl_module)

    def _generate_visualizations(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if self.viz_n < 1:
            return

        processed_dir = getattr(pl_module, "processed_dir", "data/processed")
        val_dir = Path(processed_dir) / "val"

        if not val_dir.exists():
            log.warning(f"Validation directory not found: {val_dir}")
            return

        val_slices_all = sorted(val_dir.glob("tiff*"))

        if not val_slices_all:
            log.warning("No validation slices found")
            return

        if self.selected_slice_paths is None:
            num_samples = 3 * self.viz_n
            if self.selection == "iou":
                self.selected_slice_paths = select_slices_by_iou_thirds(
                    val_slices_all, pl_module, num_samples
                )
                self.tile_rank_map = {}
            else:
                self.selected_slice_paths = val_slices_all[:num_samples]
                self.tile_rank_map = {}

            for path in self.selected_slice_paths:
                tiff_num, slice_num = parse_slice_path(path)
                self.slice_metadata[path] = (tiff_num, slice_num)

            log.info(
                f"Selected {len(self.selected_slice_paths)} validation slices for tracking"
            )

        if self._metadata_cache is None:
            self._metadata_cache = load_dataset_metadata(
                pl_module, "val", self.image_dir
            )

        output_dir = self.save_dir / "val_visualizations" if self.save_dir else None

        for slice_path in self.selected_slice_paths:
            try:
                generate_single_visualization(
                    x_path=slice_path,
                    pl_module=pl_module,
                    output_dir=output_dir,
                    epoch=trainer.current_epoch + 1,
                    title_prefix="VAL",
                    metadata_cache=self._metadata_cache,
                    image_dir=self.image_dir,
                    scale_factor=self.scale_factor,
                    tile_rank_map=self.tile_rank_map,
                )
            except Exception as e:
                import traceback

                log.error(f"Error generating visualization for {slice_path}: {e}")
                log.error(f"Full traceback: {traceback.format_exc()}")

        cleanup_gpu_memory()

        if self.save_dir:
            from glacier_mapping.utils.callback_utils import (
                log_visualizations_to_all_loggers,
            )

            val_output_dir = self.save_dir / "val_visualizations"
            log_visualizations_to_all_loggers(
                trainer, val_output_dir, trainer.current_epoch + 1, "val_visualizations"
            )


class GlacierModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, start_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = max(0, int(start_epoch))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch + 1 < self.start_epoch:
            if self.save_last:
                monitor_candidates = self._monitor_candidates(trainer)
                self._save_last_checkpoint(trainer, monitor_candidates)
            return
        super().on_train_epoch_end(trainer, pl_module)


class TestEvaluationCallback(Callback):
    """Test set evaluation triggered on best model improvement.

    Uses n-based thirds system: n=4 -> 12 visualizations (4 top + 4 middle + 4 bottom).
    """

    CLASS_NAME_TO_SHORT = {"BG": "bg", "CleanIce": "ci", "Debris": "dci"}

    def __init__(
        self,
        viz_n: int = 4,
        image_dir: str | None = None,
        scale_factor: float = 1,
        baseline_epoch: int = 15,
        aggressive_threshold: float = 0.05,
        transition_epoch: int = 50,
    ):
        super().__init__()
        self.viz_n = viz_n
        self.image_dir = Path(image_dir) if image_dir else None
        self.scale_factor = scale_factor
        self.best_test_metrics = {}
        self.baseline_epoch = baseline_epoch
        self.aggressive_threshold = aggressive_threshold
        self.transition_epoch = transition_epoch
        self.baseline_completed = False
        self.last_val_loss = float("inf")
        self._metadata_cache: Optional[Tuple[Dict, Tuple[int, int], int, Dict]] = None

    def _get_loggers(self, trainer: pl.Trainer) -> List:
        loggers: List = []
        if hasattr(trainer, "loggers") and trainer.loggers:
            loggers.extend(trainer.loggers)
        elif getattr(trainer, "logger", None):
            loggers.append(trainer.logger)
        return loggers

    def _log_metrics_to_all_loggers(
        self, trainer: pl.Trainer, metrics: Dict[str, float], step: int
    ) -> None:
        if not metrics:
            return
        for logger in self._get_loggers(trainer):
            try:
                if hasattr(logger, "log_metrics"):
                    logger.log_metrics(metrics, step=step)
                elif hasattr(logger, "experiment") and hasattr(
                    logger.experiment, "log_metrics"
                ):
                    logger.experiment.log_metrics(logger.run_id, metrics, step=step)
            except Exception as e:
                log.warning(f"Failed to log metrics to logger {logger}: {e}")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if trainer.sanity_checking:
            return
        current_val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        if current_val_loss < self.last_val_loss:
            self.last_val_loss = current_val_loss

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = None
        if checkpoint_callback is not None and hasattr(
            checkpoint_callback, "best_model_path"
        ):
            best_model_path = getattr(checkpoint_callback, "best_model_path", None)
        module_to_eval = pl_module
        if best_model_path and Path(best_model_path).exists():
            try:
                module_to_eval = pl_module.__class__.load_from_checkpoint(
                    best_model_path
                )
                module_to_eval.to(pl_module.device)
                if getattr(module_to_eval, "aryal_2023_behavior", False):
                    # Aryal unet_predict.py rebuilds the model with dropout near
                    # zero and never calls eval(); BatchNorm uses per-slice stats.
                    import torch

                    for child in module_to_eval.model.modules():
                        if isinstance(child, (torch.nn.Dropout, torch.nn.Dropout2d)):
                            child.p = 1e-8
                    module_to_eval.train()
                else:
                    module_to_eval.eval()
                log.info(
                    f"Loaded best checkpoint for final test eval: {best_model_path}"
                )
            except Exception as e:
                log.warning(f"Failed to load best checkpoint; using current model: {e}")
        self._run_full_evaluation(trainer, module_to_eval)

    def _run_full_evaluation(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is not None and hasattr(checkpoint_callback, "dirpath"):
            base_dir = Path(getattr(checkpoint_callback, "dirpath", ".")).parent
        else:
            base_dir = (
                Path(trainer.default_root_dir)
                if trainer.default_root_dir
                else Path(".")
            )
        output_dir = base_dir / "test_evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_dir = getattr(pl_module, "processed_dir", "data/processed")
        data_dir = Path(processed_dir) / "test"
        test_tiles_all = sorted(data_dir.glob("tiff*"))

        import numpy as np

        is_prebatched = False
        prebatched_X = None
        prebatched_y = None
        if not test_tiles_all:
            x_path = data_dir / "X.npy"
            y_path = data_dir / "y.npy"
            if x_path.exists() and y_path.exists():
                log.info("Using prebatched test data (X.npy / y.npy)")
                prebatched_X = np.load(x_path, mmap_mode="r")
                prebatched_y = np.load(y_path, mmap_mode="r")
                test_tiles_all = [
                    Path(data_dir / f"sample_{i:06d}") for i in range(len(prebatched_X))
                ]
                is_prebatched = True
            else:
                log.warning("No test tiles found for test evaluation")
                return

        num_samples = 3 * self.viz_n if self.viz_n > 0 else 0
        if is_prebatched:
            num_samples = 0
            test_tiles, tile_rank_map, prediction_cache = [], {}, {}
        else:
            test_tiles, tile_rank_map, prediction_cache = select_informative_test_tiles(
                test_tiles_all, pl_module, num_samples
            )

        class_names = getattr(
            pl_module,
            "class_names",
            getattr(pl_module.hparams, "class_names", ["background", "target"]),
        )
        n_classes = len(class_names)
        output_classes = getattr(pl_module, "output_classes", [1])
        rows = []
        tp_sum = [0.0] * n_classes
        fp_sum = [0.0] * n_classes
        fn_sum = [0.0] * n_classes

        from tqdm import tqdm
        import torch
        from glacier_mapping.model.evaluation import (
            IoU,
            precision,
            recall,
            tp_fp_fn,
            calculate_binary_metrics,
            predict_slice,
        )

        for idx, x_path in enumerate(tqdm(test_tiles_all, desc="Test evaluation")):
            if is_prebatched:
                x = prebatched_X[idx]
                y_true_raw = prebatched_y[idx]
                y_pred, invalid_mask = predict_slice(
                    pl_module,
                    x,
                    fill_holes=True,
                    preprocessed_chw=True,
                )
            else:
                if x_path not in prediction_cache:
                    x = np.load(x_path)
                    y_pred, invalid_mask = predict_slice(pl_module, x, fill_holes=True)
                else:
                    y_pred, invalid_mask = prediction_cache[x_path]

                y_true_raw = np.load(
                    x_path.with_name(x_path.name.replace("tiff", "mask"))
                ).astype(np.uint8)

            ignore = y_true_raw == 255
            if invalid_mask is not None:
                ignore |= invalid_mask

            row = [x_path.name]
            if len(output_classes) == 1:
                target_class = output_classes[0]
                P_target, R_target, iou_target, tp_target, fp_target, fn_target = (
                    calculate_binary_metrics(
                        y_pred, y_true_raw, target_class, mask=ignore
                    )
                )
                y_pred_bg = (y_pred == 0).astype(np.uint8)
                y_true_bg = ((y_true_raw != target_class) & (y_true_raw != 255)).astype(
                    np.uint8
                )
                P_bg, R_bg, iou_bg, tp_bg, fp_bg, fn_bg = calculate_binary_metrics(
                    y_pred_bg, y_true_bg.astype(np.uint8), target_class=1, mask=ignore
                )
                if target_class == 1:
                    tp_sum[0] += tp_bg
                    fp_sum[0] += fp_bg
                    fn_sum[0] += fn_bg
                    tp_sum[1] += tp_target
                    fp_sum[1] += fp_target
                    fn_sum[1] += fn_target
                    row += [P_bg, R_bg, iou_bg]
                    row += [P_target, R_target, iou_target]
                    row += [0.0, 0.0, 0.0]
                else:
                    tp_sum[0] += tp_bg
                    fp_sum[0] += fp_bg
                    fn_sum[0] += fn_bg
                    tp_sum[2] += tp_target
                    fp_sum[2] += fp_target
                    fn_sum[2] += fn_target
                    row += [P_bg, R_bg, iou_bg]
                    row += [0.0, 0.0, 0.0]
                    row += [P_target, R_target, iou_target]
            else:
                valid = ~ignore
                y_pred_valid = y_pred[valid]
                y_true_valid_raw = y_true_raw[valid]
                for ci in range(n_classes):
                    label = ci
                    p = (y_pred_valid == label).astype(np.uint8)
                    t = (y_true_valid_raw == label).astype(np.uint8)
                    tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p), torch.from_numpy(t))
                    tp_sum[ci] += float(tp_)
                    fp_sum[ci] += float(fp_)
                    fn_sum[ci] += float(fn_)
                    row += [
                        precision(tp_, fp_, fn_),
                        recall(tp_, fp_, fn_),
                        IoU(tp_, fp_, fn_),
                    ]
            rows.append(row)
            if (idx + 1) % 20 == 0:
                cleanup_gpu_memory(synchronize=False)

        cleanup_gpu_memory()

        csv_dir = output_dir / "csv_metrics"
        csv_dir.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        cols = ["tile"]
        for cname in class_names:
            target = self.CLASS_NAME_TO_SHORT.get(cname, cname.lower())
            cols += [f"{target}_precision", f"{target}_recall", f"{target}_iou"]

        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(csv_dir / f"epoch{trainer.current_epoch + 1:04d}.csv", index=False)

        metrics_to_log: Dict[str, float] = {}
        for ci, cname in enumerate(class_names):
            tp_, fp_, fn_ = tp_sum[ci], fp_sum[ci], fn_sum[ci]
            prec = precision(tp_, fp_, fn_)
            rec = recall(tp_, fp_, fn_)
            iou = IoU(tp_, fp_, fn_)
            target = self.CLASS_NAME_TO_SHORT.get(cname, cname.lower())
            metrics_to_log[f"full_test_{target}_precision"] = float(prec)
            metrics_to_log[f"full_test_{target}_recall"] = float(rec)
            metrics_to_log[f"full_test_{target}_iou"] = float(iou)
            if target in self.best_test_metrics:
                best_iou = self.best_test_metrics[target].get("iou", float("-inf"))
                if iou > best_iou:
                    self.best_test_metrics[target] = {
                        "iou": iou,
                        "precision": prec,
                        "recall": rec,
                    }
                    metrics_to_log[f"best_full_test_{target}_iou"] = float(iou)
                    metrics_to_log[f"best_full_test_{target}_precision"] = float(prec)
                    metrics_to_log[f"best_full_test_{target}_recall"] = float(rec)
            else:
                self.best_test_metrics[target] = {
                    "iou": iou,
                    "precision": prec,
                    "recall": rec,
                }
                metrics_to_log[f"best_full_test_{target}_iou"] = float(iou)
                metrics_to_log[f"best_full_test_{target}_precision"] = float(prec)
                metrics_to_log[f"best_full_test_{target}_recall"] = float(rec)

        # Save aggregate metrics locally so results are available without MLflow
        metrics_path = output_dir / "test_metrics.json"
        try:
            import json

            with open(metrics_path, "w") as f:
                json.dump(metrics_to_log, f, indent=2)
            log.info(f"Saved test metrics to {metrics_path}")
        except Exception as e:
            log.warning(f"Failed to save test metrics locally: {e}")

        self._log_metrics_to_all_loggers(
            trainer, metrics_to_log, step=trainer.current_epoch + 1
        )
        log_visualizations_to_all_loggers(
            trainer, output_dir, trainer.current_epoch + 1, "test_evaluations"
        )

        if self.viz_n >= 1 and num_samples >= 1:
            log.info(
                f"Generating visualizations for {min(num_samples, len(test_tiles))} tiles (n={self.viz_n})..."
            )
            self._generate_visualizations(
                pl_module,
                test_tiles[:num_samples],
                output_dir,
                trainer.current_epoch + 1,
                tile_rank_map,
                len(test_tiles_all),
                prediction_cache,
            )
            log.info("Visualizations completed.")
            cleanup_gpu_memory()
            log_visualizations_to_all_loggers(
                trainer, output_dir, trainer.current_epoch + 1, "test_evaluations"
            )

    def _generate_visualizations(
        self,
        pl_module: pl.LightningModule,
        test_tiles: List[Path],
        output_dir: Path,
        epoch: int,
        tile_rank_map: Dict[Path, int],
        total_tiles: int,
        prediction_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if self._metadata_cache is None:
            self._metadata_cache = load_dataset_metadata(
                pl_module, "test", self.image_dir
            )
        for idx, x_path in enumerate(test_tiles):
            try:
                generate_single_visualization(
                    x_path=x_path,
                    pl_module=pl_module,
                    output_dir=output_dir,
                    epoch=epoch,
                    title_prefix="TEST",
                    metadata_cache=self._metadata_cache,
                    image_dir=self.image_dir,
                    scale_factor=self.scale_factor,
                    tile_rank_map=tile_rank_map,
                )
            except Exception as e:
                log.error(f"Error generating visualization for {x_path}: {e}")
