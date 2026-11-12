import torch
import torch.nn as nn


class YOLOLoss(nn.Module):

    def __init__(
        self,
        lambda_coord=5.0,
        lambda_noobj=0.5
    ):

        super().__init__()

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, target):

        # Predictions and targets:
        # [batch, 7, 7, 5 + classes]

        pred_xy = torch.sigmoid(predictions[..., 0:2])
        pred_wh = torch.sigmoid(predictions[..., 2:4])

        pred_conf = torch.sigmoid(predictions[..., 4])

        pred_class = predictions[..., 5:]

        target_xy = target[..., 0:2]
        target_wh = target[..., 2:4]

        target_conf = target[..., 4]

        object_mask = target_conf == 1
        no_object_mask = target_conf == 0

        # 1. Bounding box loss
        xy_loss = self.mse(
            pred_xy[object_mask],
            target_xy[object_mask]
        ).sum()

        wh_loss = self.mse(
            torch.sqrt(pred_wh[object_mask] + 1e-6),
            torch.sqrt(target_wh[object_mask] + 1e-6)
        ).sum()

        box_loss = xy_loss + wh_loss

        # 2. Object confidence loss
        obj_loss = self.bce(
            pred_conf[object_mask],
            target_conf[object_mask]
        ).sum()

        # 3. No-object confidence loss
        noobj_loss = self.bce(
            pred_conf[no_object_mask],
            target_conf[no_object_mask]
        ).sum()

        # 4. Classification loss
        if object_mask.any():

            class_targets = target[..., 5:].argmax(dim=-1)

            class_loss = self.ce(
                pred_class[object_mask],
                class_targets[object_mask]
            )

        else:
            class_loss = predictions.sum() * 0.0

        total_loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        )

        return total_loss
