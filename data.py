import os
import torch
from PIL import Image
import glob


class HazeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.file_list = self.get_image_pair_list()
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        haze_image_name, ori_image_name = self.file_list[item]
        ori_image = self.transforms(Image.open(ori_image_name).convert("RGB"))
        haze_image = self.transforms(Image.open(haze_image_name).convert("RGB"))
        return ori_image, haze_image

    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        pairs = []
        all_pngs = glob.glob(os.path.join(self.root, "*.png"))
        for path in all_pngs:
            if path.endswith("_gt.png"):
                continue
            gt_path = path.replace(".png", "_gt.png")
            if os.path.exists(gt_path):
                pairs.append([path, gt_path])
        return pairs
