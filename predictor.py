import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image

def create_efficientnet_model(num_classes):
    model = EfficientNet.from_name('efficientnet-b7')  
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def load_model(model_path, create_model_func, num_classes, device):
    model = create_model_func(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  
    return image

def predict_image(image_path, model, device, size=299):
    image = preprocess_image(image_path, size).to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == '__main__':
    image_path = '/home/hy/Documents/lymphnode/dataset/dataset/elastic/1/171-3.jpg' # change your image here
    model_path = './pretrained/elastic-efficientnet_b7-19-best.pth'
    num_classes = 3 
    device = torch.device("cpu")
    model = load_model(model_path, create_efficientnet_model, num_classes, device)
    prediction = predict_image(image_path, model, device, size=299)
    print(f"EfficientNet-B7 prediction: {prediction}")
