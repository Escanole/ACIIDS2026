import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file, base_path, transform=None, policy="ones"):
        """
        image_list_file: path to CSV file (train/val/test)
        base_path: root folder where `train/`, `valid/`, `test/` folders exist
        """
        self.base_path = base_path
        df = pd.read_csv(image_list_file)

        self.label_columns = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
            "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]

        # Handle uncertain labels
        if policy == "ones":
            df[self.label_columns] = df[self.label_columns].replace(-1, 1)
        elif policy == "zeroes":
            df[self.label_columns] = df[self.label_columns].replace(-1, 0)
        df[self.label_columns] = df[self.label_columns].fillna(0)

        # Store relative paths and labels
        self.image_names = df['Path'].tolist()
        self.labels = df[self.label_columns].astype(float).values.tolist()
        self.transform = transform

    def __getitem__(self, index):
        relative_path = self.image_names[index]
    
        # Updated path fixing logic to use base_path from config
        if relative_path.startswith("train/") or "CheXpert-v1.0-small/train" in relative_path:
            fixed_path = relative_path.split("train/")[-1]
            full_path = os.path.join(self.base_path, "train", fixed_path)
        elif relative_path.startswith("valid/") or "CheXpert-v1.0-small/valid" in relative_path:
            fixed_path = relative_path.split("valid/")[-1]
            full_path = os.path.join(self.base_path, "valid", fixed_path)
        elif relative_path.startswith("test/") or "CheXpert/test" in relative_path:
            fixed_path = relative_path.split("test/")[-1]
            full_path = os.path.join(self.base_path, "test", fixed_path)
        else:
            # Fallback: assume path is relative to base_path
            full_path = os.path.join(self.base_path, relative_path)
    
        if not os.path.exists(full_path):
            print(f"[WARN] Missing file: {full_path}")
            return self.__getitem__((index + 1) % len(self))
    
        # Load image
        image = Image.open(full_path).convert('RGB')
        label = self.labels[index]
    
        if self.transform is not None:
            image = self.transform(image)
    
        # Extract task and domain labels for DANN
        task_label = torch.FloatTensor(label[:-1])     # All disease labels except support device
        support_device_label = label[-1]               # 1.0 if device present, 0.0 if not
        domain_label = torch.tensor([support_device_label], dtype=torch.float32)
    
        return image, task_label, domain_label, self.image_names[index]

    def __len__(self):
        return len(self.image_names)
