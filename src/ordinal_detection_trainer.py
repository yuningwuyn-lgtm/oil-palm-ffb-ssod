"""Ultralytics YOLO detection trainer with an ordinal maturity auxiliary loss.

The detector still predicts the canonical six classes.  Only the four ordered
maturity classes participate in the auxiliary term:

    unripe < under_ripe < ripe < overripe

The additional loss is evaluated on foreground anchors after task-aligned
assignment.  Abnormal and empty bunches remain ordinary detection classes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class OrdinalDetectionLoss(v8DetectionLoss):
    """YOLOv8 detection loss with a foreground ordinal maturity penalty."""

    # Canonical schema: abnormal=0, empty=1, overripe=2, ripe=3,
    # under_ripe=4, unripe=5.
    ordinal_class_ids = (5, 4, 3, 2)

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.ordinal_gain = float(getattr(model, "ordinal_loss_gain", 0.15))

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple:
        """Compute standard detection loss plus a rank-aware classification term."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
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
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.any() and self.ordinal_gain > 0:
            ordered_ids = torch.tensor(self.ordinal_class_ids, device=self.device, dtype=torch.long)
            rank_values = torch.arange(len(self.ordinal_class_ids), device=self.device, dtype=pred_scores.dtype)
            fg_scores = pred_scores[fg_mask]
            fg_targets = target_scores[fg_mask]
            fg_labels = fg_targets.argmax(dim=-1)
            ordinal_mask = (fg_labels[..., None] == ordered_ids).any(dim=-1)
            if ordinal_mask.any():
                ordered_prob = fg_scores[ordinal_mask][:, ordered_ids].sigmoid()
                expected_rank = (ordered_prob * rank_values).sum(dim=-1) / ordered_prob.sum(dim=-1).clamp_min(1e-6)
                target_rank = (fg_labels[ordinal_mask, None] == ordered_ids).to(dtype).argmax(dim=-1).to(dtype)
                anchor_weight = fg_targets[ordinal_mask].amax(dim=-1).detach()
                ordinal_loss = (F.smooth_l1_loss(expected_rank, target_rank, reduction="none") * anchor_weight).sum()
                ordinal_loss /= anchor_weight.sum().clamp_min(1.0)
                loss[1] += self.ordinal_gain * ordinal_loss

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

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()


class OrdinalDetectionModel(DetectionModel):
    """Detection model that instantiates :class:`OrdinalDetectionLoss`."""

    ordinal_loss_gain = 0.15

    def init_criterion(self):
        return OrdinalDetectionLoss(self)


def make_ordinal_trainer(ordinal_loss_gain: float = 0.15):
    """Return an Ultralytics trainer class configured for one ablation gain."""

    class ConfiguredOrdinalDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
            model = OrdinalDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose,
            )
            model.ordinal_loss_gain = ordinal_loss_gain
            if weights:
                model.load(weights)
            return model

    return ConfiguredOrdinalDetectionTrainer


