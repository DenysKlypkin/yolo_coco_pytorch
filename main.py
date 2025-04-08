from pathlib import Path

import pandas as pd
import os
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path

import torch


dataDir = Path("datasets/coco/images/val2017")
annFile = Path("datasets/coco/annotations/instances_val2017.json")
coco = COCO(annFile)
imgIds = coco.getImgIds()

len(coco.imgs)

coco.imgs[2]


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


# for ann in anns:
#     bbox = ann["bbox"]
#     bboxes.append(bbox)

# # get classes for each bbox
# classes = []
# annIds = coco.getAnnIds(imgIds=[img_info["id"]])
# anns = coco.loadAnns(annIds)
# for ann in anns:
#     classes.append(ann["category_id"])
