import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

class FLDataset(Dataset):
    def __init__(self, root_dir, train=False):
        self.root_dir = root_dir
        self.train = train
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        folders = sorted(os.listdir(self.root_dir))
        image_paths = []
        labels = []
        for folder in folders:
            folder_path = os.path.join(self.root_dir, folder)
            for filename in os.listdir(folder_path):
                image_path = self._get_image_path(folder_path, filename)
                if image_path:
                    image_paths.append(image_path)
                    labels.append(int(filename.split('_')[0]))
                    break  # Remove this line if you want to load all images in the folder
        return image_paths, labels

    def _get_image_path(self, folder_path, filename):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            return os.path.join(folder_path, filename)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        return image, label

class TripletDataset(Dataset):
    def __init__(self, dataset, transform=None, crop_transform=None, crop_coef=0.4, is_triplet=True):
        self.dataset = dataset
        self.transform = transform
        self.crop_transform = crop_transform
        self.crop_coef = crop_coef
        self.is_triplet = is_triplet
        self.class_indices = self._get_class_indices()

    def _get_class_indices(self):
        class_indices = {}
        for idx, label in enumerate(self.dataset.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor_image, anchor_label = self.dataset[idx]

        positive_image = self._get_positive_image(anchor_image)
        
        if self.is_triplet:
            negative_image = self._get_negative_image(anchor_label)
            return (self.transform(anchor_image), 
                    self.transform(positive_image), 
                    self.transform(negative_image))
        else:
            return (self.transform(anchor_image), 
                    self.transform(positive_image))

    def _get_positive_image(self, anchor_image):
        width, height = anchor_image.size
        crop_transform = v2.Compose([
            v2.RandomResizedCrop((int(width * self.crop_coef), int(height * self.crop_coef)))
        ])
        if self.crop_transform:
            return self.crop_transform(anchor_image)
        else:
            return crop_transform(anchor_image)

    def _get_negative_image(self, anchor_label):
        negative_label = random.choice([label for label in self.class_indices if label != anchor_label])
        negative_idx = random.choice(self.class_indices[negative_label])
        negative_image, _ = self.dataset[negative_idx]
        return negative_image
