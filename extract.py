import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from timm import create_model
import cv2
import numpy as np
from PIL import Image

def create_resnet_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_efficientnet_model(num_classes):
    model = EfficientNet.from_name('efficientnet-b7')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def create_convnext_model(num_classes):
    model = create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
    return model

def create_vit_model(num_classes):
    model = create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    return model

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = cv2.imread(img_path, 1)[:, :, ::-1]  # BGR to RGB
    image = cv2.resize(image, (299, 299))
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

def visualize_cam(model, img_path, device):
    target_layers = [model._blocks[-1]]  # Last block of EfficientNet-B7
    cam = GradCAM(model=model, target_layers=target_layers)

    raw_image, input_tensor = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)
    targets = None  # Use the default target
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(np.array(raw_image) / 255.0, grayscale_cam, use_rgb=True)
    
    return visualization

def load_model(model_path, create_model_func, num_classes, device):
    model = create_model_func(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def create_collage(image_list, collage_name, images_per_row=25):
    images = [Image.fromarray(img) for img in image_list]
    widths, heights = zip(*(i.size for i in images))

    max_height = max(heights)
    total_width = sum(widths[:images_per_row])
    num_rows = (len(images) + images_per_row - 1) // images_per_row

    collage_image = Image.new('RGB', (total_width, max_height * num_rows))

    x_offset = 0
    y_offset = 0

    for i, img in enumerate(images):
        collage_image.paste(img, (x_offset, y_offset))
        x_offset += img.size[0]
        if (i + 1) % images_per_row == 0:
            x_offset = 0
            y_offset += img.size[1]

    collage_image.save(collage_name)

if __name__ == "__main__":
    for type_ in [0,1,2]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model('./pretrained/elastic-efficientnet_b7.pth', create_efficientnet_model, 3, device)
        img_folder = './dataset/elastic/{}/'.format(type_)
        output_folder = './extract/gradcam_output_{}/'.format(type_)
        os.makedirs(output_folder, exist_ok=True)

        image_list = []
        for img_file in os.listdir(img_folder):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(img_folder, img_file)
                visualization = visualize_cam(model, img_path, device)
                image_list.append(visualization)
                cv2.imwrite(os.path.join(output_folder, f'gradcam_{img_file}'), visualization)

        create_collage(image_list, './extract/total/gradcam_collage_{}.png'.format(type_))
