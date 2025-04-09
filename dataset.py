import io
import torch
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO

from config import DATA_DIR


class COCODataset(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.B = 5  # number of bounding boxes per grid cell
        self.S = 7  # number of grid cells
        self.C = len(self.coco.getCatIds())

        self.classes = self.coco.loadCats(self.coco.getCatIds())
        self.clsids2clsnames = {cls["id"]: cls["name"] for cls in self.classes}
        self.clsnames2clsids = {cls["name"]: cls["id"] for cls in self.classes}

    def __len__(self):
        return len(self.coco.imgs)

    def __getitem__(self, idx):
        print(f"Loading image {idx}...")
        imgIds = self.coco.getImgIds()
        img_info = self.coco.loadImgs(imgIds[idx])[0]

        annIds = self.coco.getAnnIds(imgIds=[img_info["id"]])
        anns = self.coco.loadAnns(annIds)

        bboxes = [ann["bbox"] for ann in anns]
        classes = [ann["category_id"] for ann in anns]
        img = io.imread(DATA_DIR / img_info["file_name"])

        im_tensor = torch.from_numpy(img / 255.0).permute(2, 1, 0).float()
        w, h = im_tensor.shape[1:3]
        # ? need to normilize by channels

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int8)
        w, h = im_tensor.shape[:2]
        target_dim = 5 * self.B + self.C

        targets = torch.zeros(self.S, self.S, target_dim)

        cell_width = w // self.S
        cell_height = h // self.S

        box_widths = bboxes[:, 2]
        box_heights = bboxes[:, 3]
        box_center_xs = bboxes[:, 0] + 0.5 * bboxes[:, 2]
        box_center_ys = bboxes[:, 1] + 0.5 * bboxes[:, 3]

        # i, j indexes of the grid cell that contains the center of the bounding box
        box_i = torch.floor(box_center_xs / cell_width).long()
        box_j = torch.floor(box_center_ys / cell_height).long()

        # the offset of the center for each bounding box
        box_xc_cell_offset = (box_center_xs - box_i * cell_width) / cell_width
        box_yc_cell_offset = (box_center_ys - box_j * cell_height) / cell_height

        box_w_label = box_widths / w
        box_h_label = box_heights / h

        for idx, b in enumerate(range(bboxes.size(0))):
            for k in range(self.B):
                s = k * 5
                targets[box_i[idx], box_j[idx], s] = box_xc_cell_offset[idx]
                targets[box_i[idx], box_j[idx], s + 1] = box_yc_cell_offset[idx]
                targets[box_i[idx], box_j[idx], s + 2] = box_w_label[idx].sqrt()
                targets[box_i[idx], box_j[idx], s + 3] = box_h_label[idx].sqrt()
                targets[box_i[idx], box_j[idx], s + 4] = 1.0  # confidence score
            targets[box_i[idx], box_j[idx], self.B * 5 + classes[idx]] = 1.0

        if len(bboxes) > 0:
            bboxes /= torch.Tensor([[w, h, w, h]]).expand_as(bboxes)

        targets = {
            "bboxes": bboxes,
            "classes": classes,
            "target": targets,
        }
        return img, targets
