import torch
import torchvision.transforms as transforms


from count_parameters import count_param
from models.inception_resnet_v1 import InceptionResnetV1
from models.ModifiedInceptionResnetV1 import ModifiedInceptionResnetV1
from TransformDataset import TripletDataset
import argparse


def validation(model, loader, device='cpu'):
    # Предобработка данных
    aligned = []
    aligned2 = []

    for x, y in loader: 
        x = transform_resize(x)
        aligned.append(transform(x))
        aligned2.append(transform_crop(x))
    
    # Запись тензоров для больших изображений
    if args.save_tensor == 'True':
        embeddings = []
        for index in range(len(aligned)):
            print(f'save tensor {index}')
            res = model(aligned[index].unsqueeze(0).to(device)).detach().cpu()
            embeddings.append(res)
        torch.save(torch.stack(embeddings), str(args.filename))


    embedding = torch.load(str(args.filename))
    embedding = embedding.squeeze(1)

    true_answer = 0
    false_answer = 0

    # Проход по каждому кропнутому тензору и косинусовое сходство со всеми другими тензорами
    for index in range(len(aligned2)):
        embedding_crop = model(aligned2[index].unsqueeze(0).to(device)).detach().cpu()
        distance = torch.nn.functional.cosine_similarity(embedding_crop, embedding.to())
        most_similar_cosine_indices = torch.argsort(distance, descending=True).squeeze(0)
        
        print(f'cosine similarity:{index} / most similar:{most_similar_cosine_indices[0:6]}')
        if most_similar_cosine_indices[0] == index:
            true_answer +=1
        else:
            false_answer +=1


    print(f"True: {true_answer}, False: {false_answer}")
    accuracy = float(true_answer / len(loader)) * 100
    print(f"accuracy {accuracy:.4f}%")




parser = argparse.ArgumentParser(description="Сохранение тензора")
parser.add_argument("-save_tensor", nargs='?', default='True', help="Сохранения не кропнутых тензоров в файл")
parser.add_argument("-filename", type=str, default='tensor.pt', help="Название файла для сохранения")
args = parser.parse_args()

#Data_load
loader = TripletDataset('../FakeDataset/test')

#transforms
transform_resize = transforms.Compose([
    transforms.Resize((641, 480)),
])
# Обрезание изображия из исходного
transform_crop = transforms.Compose([
    transforms.RandomCrop(size=(350, 350)),
    # transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Применяется для исходного изображения 
transform = transforms.Compose([
    transforms.ToTensor(), # Преобразование изображения в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Определение устройства для выполнения вычислений
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))



# model = InceptionResnetV1(pretrained='casia-webface', device=device)
model_path = "model_weights4.pt"
model = ModifiedInceptionResnetV1(device=device)
model.load_state_dict(torch.load(model_path))

model.eval()

# Вывод количества параметров
count_param(model, show_layers=False)

validation(model, loader, device=device)