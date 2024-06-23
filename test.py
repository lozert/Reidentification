import torch
import torchvision.transforms as transforms
from count_parameters import count_param
from models.inception_resnet_v1 import InceptionResnetV1
from models.ModifiedInceptionResnetV1 import ModifiedInceptionResnetV1
from TransformDataset import FLDataset
import argparse

# Глобальные переменные для сохраненных тензоров


def save_tensors(model, data_loader, device, resize_transform, crop_transform, full_transform):
    embeddings_full = {}
    embeddings_crop = {}

    for img, label in data_loader:
        resized_img = resize_transform(img)
        crop_embedding = model(crop_transform(resized_img).unsqueeze(0).to(device)).detach().cpu()
        full_embedding = model(full_transform(resized_img).unsqueeze(0).to(device)).detach().cpu()
        
        embeddings_crop[label] = crop_embedding
        embeddings_full[label] = full_embedding

    torch.save(embeddings_crop, 'dict_crop_tensor.pt')
    torch.save(embeddings_full, 'dict_tensor.pt')
    print('Save tensor done')

def validate_model(model, data_loader, device, resize_transform, crop_transform, full_transform):

    if args.save_tensor:
        save_tensors(model, data_loader, device, resize_transform, crop_transform, full_transform)

    embeddings_full = torch.load('dict_tensor.pt')
    embeddings_crop = torch.load('dict_crop_tensor.pt')

    correct_predictions = 0
    incorrect_predictions = 0

    for crop_label, crop_embedding in embeddings_crop.items():
        highest_similarity = -10
        predicted_label = None

        for full_label, full_embedding in embeddings_full.items():
            cosine_similarity = torch.nn.functional.cosine_similarity(crop_embedding, full_embedding)
            if cosine_similarity > highest_similarity:
                highest_similarity = cosine_similarity
                predicted_label = full_label

        print(f"Predicted label: {predicted_label} / Actual label: {crop_label}")
        if predicted_label == crop_label:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    accuracy = (correct_predictions / len(data_loader)) * 100
    print(f"Correct: {correct_predictions}, Incorrect: {incorrect_predictions}")
    print(f"Accuracy: {accuracy:.4f}%")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tensor saving script")
    parser.add_argument("--save_tensor", action='store_true', help="Save cropped and full tensors to file")
    parser.add_argument("--filename", type=str, default='tensor.pt', help="Filename for saving tensors")
    return parser.parse_args()

def main():

    # Data loader
    data_loader = FLDataset('../FakeDataset/test')

    # Transforms
    resize_transform = transforms.Compose([transforms.Resize((641, 480))])
    crop_transform = transforms.Compose([
        transforms.RandomCrop(size=(350, 350)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # Model
    model_path = "model_weights2.pt"
    model = ModifiedInceptionResnetV1(device=device, num_classes=516)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Print number of parameters
    count_param(model, show_layers=False)

    # Validation
    validate_model(model, data_loader, device, resize_transform, crop_transform, full_transform)

args = parse_arguments()

if __name__ == "__main__":
    main()
