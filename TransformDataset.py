from torchvision.transforms import v2

from torch.utils.data import Dataset
import os
from PIL import Image


class LFWFacesDataset(Dataset):
    def __init__(self, root_dir, train=False, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        selected_folders = sorted(os.listdir(self.root_dir))
     
        image_paths = []
        labels = []
        for i, folder in enumerate(selected_folders):
            folder_path = os.path.join(self.root_dir, folder)
            for filename in os.listdir(folder_path):
                image, label = self._image_label(folder_path, filename)
                image_paths.append(image)
                labels.append(i)
                break
        return image_paths, labels

    def _image_label(self, folder_path, filename):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) :
            return os.path.join(folder_path, filename), filename
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label



import random

crop_transform = v2.Compose([
    v2.RandomResizedCrop((300, 300))
])

class TripletDataset(Dataset):
    def __init__(self, dataset,  transform=None):
        self.dataset = dataset
        self.transform = transform
        self.class_indices = {}
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor_label = self.dataset.labels[idx]

        anchor_img = self.dataset[anchor_label][0]
        positive_img = crop_transform(anchor_img)

        negative_label = random.choice([label for label in self.class_indices.keys() if label != anchor_label])
        negative_idx = self.class_indices[negative_label][0]
        negative_img = self.dataset[negative_idx][0]
        
        return self.transform(anchor_img), self.transform(positive_img), self.transform(negative_img)