from pathlib import Path

import pandas as pd
import os
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn

import torchvision
import torch

from train import train


train()

dataDir = Path("datasets/coco/images/val2017")
annFile = Path("datasets/coco/annotations/instances_val2017.json")
coco = COCO(annFile)
imgIds = coco.getImgIds()

len(coco.imgs)


len(imgIds)

len(coco.getCatIds())
imgs = coco.loadImgs(imgIds[3])


# coco get bboxes
bboxes = []
for img in imgs:
    annIds = coco.getAnnIds(imgIds=[img["id"]])
    anns = coco.loadAnns(annIds)
    for ann in anns:
        bbox = ann["bbox"]
        bboxes.append(bbox)

# get classes for each bbox
classes = []
for img in imgs:
    annIds = coco.getAnnIds(imgIds=[img["id"]])
    anns = coco.loadAnns(annIds)
    for ann in anns:
        classes.append(ann["category_id"])


# _, axs = plt.subplots(len(imgs), 2, figsize=(10, 5 * len(imgs)))
# for img, ax in zip(imgs, axs):
#     I = io.imread(dataDir / img["file_name"])
#     annIds = coco.getAnnIds(imgIds=[img["id"]])
#     anns = coco.loadAnns(annIds)
#     ax[0].imshow(I)
#     ax[1].imshow(I)
#     plt.sca(ax[1])
#     coco.showAnns(anns, draw_bbox=False)


imgIds = coco.getImgIds()
img_info = coco.loadImgs(imgIds[2])[0]
bboxes = []
annIds = coco.getAnnIds(imgIds=[img_info["id"]])
anns = coco.loadAnns(annIds)

bboxes = [ann["bbox"] for ann in anns]
classes = [ann["category_id"] for ann in anns]

I = io.imread(dataDir / img["file_name"])

I.shape
img = coco.loadImgs(imgIds[2])[0]

im_tensor = torch.from_numpy(I / 255.0).permute(2, 1, 0).float()
w, h = im_tensor.shape[1:3]


cell_width = w // 7
cell_height = h // 7

bboxes = torch.tensor(bboxes, dtype=torch.float32)
classes = torch.tensor(classes, dtype=torch.int8)

box_widths = bboxes[:, 2]
box_heights = bboxes[:, 3]
box_center_xs = bboxes[:, 0] + 0.5 * bboxes[:, 2]
box_center_ys = bboxes[:, 1] + 0.5 * bboxes[:, 3]


# i, j indexes of the grid cell that contains the center of the bounding box
box_i = torch.floor(box_center_xs / cell_width).long()
box_j = torch.floor(box_center_ys / cell_height).long()

classes = coco.loadCats(coco.getCatIds())
clsids2clsnames = {cls["id"]: cls["name"] for cls in classes}
clsnames2clsids = {cls["name"]: cls["id"] for cls in classes}


for idx, b in enumerate(range(bboxes.size(0))):
    print(idx, "idx")
    print(b, "b")


if len(bboxes) > 0:
    bboxes /= torch.Tensor([[w, h, w, h]]).expand_as(bboxes)


backbone = torchvision.models.resnet18(pretrained=True)

backbone = nn.Sequential(
    *list(backbone.children())[:-2]  # Remove the last two layers (avgpool and fc)
)


shifts_x = torch.arange(0, 7, dtype=torch.int32) * 1 / float(7)
shifts_y = (
    torch.arange(
        0,
        7,
        dtype=torch.int32,
    )
    * 1
    / float(7)
)
# it is like cartesian product of the 2 tensors
shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

shifts_x.view((1, 7, 7, 1))


shifts_x = shifts_x.view((1, 7, 7, 1)).repeat(1, 1, 1, 7)
shifts_y = shifts_y.view((1, 7, 7, 1)).repeat(1, 1, 1, 7)


# for ann in anns:
#     bbox = ann["bbox"]
#     bboxes.append(bbox)

# # get classes for each bbox
# classes = []
# annIds = coco.getAnnIds(imgIds=[img_info["id"]])
# anns = coco.loadAnns(annIds)
# for ann in anns:
#     classes.append(ann["category_id"])
