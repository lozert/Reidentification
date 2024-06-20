import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class LFWFacesDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        # Получаем список всех папок (имена людей)
        selected_folders = sorted(os.listdir(self.root_dir))
        # Загружаем пути к изображениям и соответствующие им метки
        image_paths = []
        labels = []
        for i, folder in enumerate(selected_folders):
            folder_path = os.path.join(self.root_dir, folder)
            for filename in os.listdir(folder_path):
                
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) and filename != '.DS_Store':
                    image_paths.append(os.path.join(folder_path, filename))
                    labels.append(i)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Преобразования для тренировочных и валидационных данных
train_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Применяется для исходного изображения 
val_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(), # Преобразование изображения в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Создаем объекты класса датасета для тренировочных и валидационных данных
train_dataset = LFWFacesDataset('../FakeDataset/train', train=True, transform=train_transform)
val_dataset = LFWFacesDataset('../FakeDataset/test', train=False, transform=val_transform)


import torch    
from torch.utils.data import Dataset, DataLoader

class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_indices = {}
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor_label = self.dataset.labels[idx]

        # Select positive sample from the same class
        positive_idx, anchor_idx = random.sample(self.class_indices[anchor_label],2)
        positive_img = self.dataset[positive_idx][0]
        anchor_img = self.dataset[anchor_idx][0]

        # Select negative sample from a different class
        negative_label = random.choice([label for label in self.class_indices.keys() if label != anchor_label])
        negative_idx = random.choice(self.class_indices[negative_label])
        negative_img = self.dataset[negative_idx][0]

        return anchor_img, positive_img, negative_img


train_triplet_dataset = TripletDataset(train_dataset)
val_triplet_dataset = TripletDataset(val_dataset)

batch_size = 64
train_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_triplet_dataset, batch_size=batch_size, shuffle=False)



    
#Загрузка модели
from models.ModifiedInceptionResnetV1 import ModifiedInceptionResnetV1
import torch

# Определение устройства для выполнения вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "model_weights.pth"
model = ModifiedInceptionResnetV1(device='cuda')
model.load_state_dict(torch.load(model_path))

for name, param in model.named_parameters():
    if name.startswith('last')  or name.startswith('block8'): 
        param.requires_grad = True
    else:
        param.requires_grad = False

print('Пока код не рабочий')

## TODO: Идёт работа над кодом
# from models.utils.training import pass_epoch

# #Лосс
# loss_fn = nn.TripletMarginLoss()

# # Оптимизатор
# optimizer = optim.Adam(model.parameters(), lr=0.005)

# num_epochs = 2

# for epoch in range(num_epochs):
#     print("----------------")
#     print(f"Epoch {epoch+1}/{num_epochs} training:")
#     model.train()
#     loss, metric = pass_epoch(model,  loss_fn, train_loader, optimizer, device=device)
#     print(f"Loss: {loss} \n Metric: {metric}" )
    
# x = int(input("Сохранение модели 0-да, 1-нет"))
# if x == 0:
#     model.eval()
#     # Сохранение весов модели в формате .pth
#     torch.save(model.state_dict(), 'model_weights.pth')