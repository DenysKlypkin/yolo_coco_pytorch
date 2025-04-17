import torch
import torch.nn as nn


def get_IOU(boxes_1: torch.Tensor, boxes_2: torch.Tensor):
    # boxes : [x_center, y_center, width, height]

    # top corner left of the 2 boxes
    top_left_x = torch.max(
        boxes_1[:, 0] - boxes_1[:, 2] / 2, boxes_2[:, 0] - boxes_2[:, 2] / 2
    )
    top_left_y = torch.max(
        boxes_1[:, 1] - boxes_1[:, 3] / 2, boxes_2[:, 1] - boxes_2[:, 3] / 2
    )

    # bottom corner right of the 2 boxes
    bottom_right_x = torch.min(
        boxes_1[:, 0] + boxes_1[:, 2] / 2, boxes_2[:, 0] + boxes_2[:, 2] / 2
    )
    bottom_right_y = torch.min(
        boxes_1[:, 1] + boxes_1[:, 3] / 2, boxes_2[:, 1] + boxes_2[:, 3] / 2
    )
    # intersection areas

    intersection_area = torch.clamp(bottom_right_x - top_left_x, min=0) * torch.clamp(
        bottom_right_y - top_left_y, min=0
    )

    # areas of the 2 boxes
    boxes_1_area = boxes_1[:, 2] * boxes_1[:, 3]
    boxes_2_area = boxes_2[:, 2] * boxes_2[:, 3]

    # union areas
    union_area = boxes_1_area + boxes_2_area - intersection_area

    # IOU
    iou = intersection_area / union_area + 1e-6
    return iou


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=5, C=80, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        batch_size = preds.size(0)
        print(batch_size, "preds")
        print(preds.shape, "preds")
        preds = preds.view(batch_size, self.S, self.S, self.B * (5 + self.C))

        shifts_x = torch.arange(0, self.S, dtype=torch.int32) * 1 / float(self.S)
        shifts_y = (
            torch.arange(
                0,
                self.S,
                dtype=torch.int32,
            )
            * 1
            / float(self.S)
        )
        # i have to add them to lsat dim of (batch, s,s,5,b)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shifts_x = (
            shifts_x.view((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B).to("cuda")
        )
        shifts_y = (
            shifts_y.view((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B).to("cuda")
        )

        pred_boxes = preds[..., : self.B * 5].view(
            batch_size, self.S, self.S, self.B, 5
        )
        # from pred_boxes of (x_offset, y_offset, w, h)
        # to calculate the IOU, i need to convert the boxes to (x_center, y_center, w, h)

        pred_boxes_x_center = pred_boxes[..., 0] + shifts_x
        pred_boxes_y_center = pred_boxes[..., 1] + shifts_y
        pred_boxes_w = pred_boxes[..., 2].square()
        pred_boxes_h = pred_boxes[..., 3].square()
        pred_boxes = torch.cat(
            [pred_boxes_x_center, pred_boxes_y_center, pred_boxes_w, pred_boxes_h],
            dim=-1,
        )

        target_boxes = targets[..., : self.B * 5].view(
            batch_size, self.S, self.S, self.B, 5
        )
        target_boxes_x_center = target_boxes[..., 0] + shifts_x
        target_boxes_y_center = target_boxes[..., 1] + shifts_y
        target_boxes_w = target_boxes[..., 2].square()
        target_boxes_h = target_boxes[..., 3].square()
        target_boxes = torch.cat(
            [
                target_boxes_x_center,
                target_boxes_y_center,
                target_boxes_w,
                target_boxes_h,
            ],
            dim=-1,
        )

        iou = get_IOU(pred_boxes, target_boxes)

        max_iou_val, max_iou_idx = iou.max(dim=-1, keepdim=True)
        max_iou_idx = max_iou_idx.repeat(1, 1, 1, self.B)

        bb_idxs = (
            torch.arange(self.B)
            .reshape(1, 1, 1, self.B)
            .expand_as(max_iou_idx)
            .to("cuda")
        )
        is_max_iou_box = (max_iou_idx == bb_idxs).long()
        obj_indicator = targets[..., 4:5]

        #######################
        # Classification Loss #
        #######################
        cls_target = targets[..., 5 * self.B :]
        cls_preds = preds[..., 5 * self.B :]
        cls_mse = (cls_preds - cls_target) ** 2
        cls_mse = (obj_indicator * cls_mse).sum()

        ######################################################
        # Objectness Loss (For responsible predictor boxes ) #
        ######################################################
        is_max_box_obj_indicator = is_max_iou_box * obj_indicator
        obj_mse = (pred_boxes[..., 4] - max_iou_val) ** 2
        obj_mse = (is_max_box_obj_indicator * obj_mse).sum()

        #####################
        # Localization Loss #
        #####################
        x_mse = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2
        x_mse = (is_max_box_obj_indicator * x_mse).sum()

        y_mse = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        y_mse = (is_max_box_obj_indicator * y_mse).sum()
        w_sqrt_mse = (pred_boxes[..., 2] - target_boxes[..., 2]) ** 2
        w_sqrt_mse = (is_max_box_obj_indicator * w_sqrt_mse).sum()
        h_sqrt_mse = (pred_boxes[..., 3] - target_boxes[..., 3]) ** 2
        h_sqrt_mse = (is_max_box_obj_indicator * h_sqrt_mse).sum()

        #################################################
        # Objectness Loss
        #################################################
        no_object_indicator = 1 - is_max_box_obj_indicator
        no_obj_mse = (pred_boxes[..., 4] - torch.zeros_like(pred_boxes[..., 4])) ** 2
        no_obj_mse = (no_object_indicator * no_obj_mse).sum()

        loss = self.lambda_coord * (x_mse + y_mse + w_sqrt_mse + h_sqrt_mse)
        loss += cls_mse + obj_mse
        loss += self.lambda_noobj * no_obj_mse
        loss = loss / batch_size
        return loss
