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

    def __len__(self):
        return len(self.coco.imgs)

    def __getitem__(self, idx):
        imgIds = self.coco.getImgIds()
        img_info = self.coco.loadImgs(imgIds[idx])[0]

        annIds = self.coco.getAnnIds(imgIds=[img_info["id"]])
        anns = self.coco.loadAnns(annIds)

        bboxes = [ann["bbox"] for ann in anns]
        classes = [ann["category_id"] for ann in anns]
        img = io.imread(DATA_DIR / img_info["file_name"])

        im_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()

        # ? need to normilize by channels

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int8)

        target_dim = 5 * self.B + self.C
