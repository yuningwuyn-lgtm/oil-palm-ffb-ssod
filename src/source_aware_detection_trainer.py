"""Ultralytics detection trainer with active per-source classification loss masks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def load_prefix_masks(dataset_root: Path, nc: int) -> dict[int, list[float]]:
    """Load observed source masks written by the merge step."""
    metadata_path = dataset_root / "source_metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    masks: dict[int, list[float]] = {}
    for index, source in enumerate(metadata.get("sources", []), start=1):
        source_mask = source.get("observed_class_loss_mask") or source.get("class_loss_mask", {})
        canonical = metadata.get("canonical_classes", {})
        masks[index] = [
            float(source_mask.get(str(canonical.get(str(class_id), canonical.get(class_id, class_id))), 1))
            for class_id in range(nc)
        ]
    return masks


class SourceAwareDetectionLoss(v8DetectionLoss):
    """Mask BCE classification gradients for classes absent from each labeled source."""

    def hyp_value(self, name: str) -> float:
        defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        if isinstance(self.hyp, dict):
            return float(self.hyp.get(name, defaults[name]))
        return float(getattr(self.hyp, name, defaults[name]))

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        source_mask = batch.get("source_class_loss_mask")
        if source_mask is not None:
            if source_mask.shape[-1] != pred_scores.shape[-1]:
                raise ValueError(
                    f"source_class_loss_mask nc={source_mask.shape[-1]} does not match model nc={pred_scores.shape[-1]}"
                )
            bce_loss *= source_mask.to(device=self.device, dtype=dtype).unsqueeze(1)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        loss[0] *= self.hyp_value("box")
        loss[1] *= self.hyp_value("cls")
        loss[2] *= self.hyp_value("dfl")
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()


class SourceAwareDetectionModel(DetectionModel):
    def init_criterion(self):
        return SourceAwareDetectionLoss(self)


def make_source_aware_trainer():
    class ConfiguredSourceAwareDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
            model = SourceAwareDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
            if weights:
                model.load(weights)
            return model

        def preprocess_batch(self, batch: dict) -> dict:
            batch = super().preprocess_batch(batch)
            dataset_root = Path(self.data["path"])
            masks = load_prefix_masks(dataset_root, int(self.data["nc"]))
            default_mask = [1.0] * int(self.data["nc"])
            rows = []
            for image_path in batch.get("im_file", []):
                match = re.match(r"src(\d+)_", Path(image_path).name, flags=re.IGNORECASE)
                rows.append(masks.get(int(match.group(1)), default_mask) if match else default_mask)
            batch["source_class_loss_mask"] = torch.tensor(rows, device=self.device, dtype=torch.float32)
            return batch

    return ConfiguredSourceAwareDetectionTrainer

