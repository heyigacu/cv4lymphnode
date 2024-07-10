import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Define the AdaRes Network
class AdaNorm(nn.Module):
    def __init__(self, num_features):
        super(AdaNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        return self.scale * self.bn(x) + (1 - self.scale) * x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.adanorm = AdaNorm(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.adanorm(out)
        return out + residual

class AdaRes(nn.Module):
    def __init__(self):
        super(AdaRes, self).__init__()
        self.conv1 = nn.Conv2d(1, 850, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(850, 850, kernel_size=3, padding=1),
            nn.ReLU(),
            AdaNorm(850)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(850, 850, kernel_size=3, padding=1),
            nn.ReLU(),
            AdaNorm(850)
        )

        self.resblock1 = ResBlock(850, 850)
        self.resblock2 = ResBlock(850, 850)
        
        self.final_conv = nn.Conv2d(850, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        
        x = branch1 + branch2
        
        x = self.resblock1(x)
        x = self.resblock2(x)
        
        x = self.final_conv(x)
        return x

def create_adares_model():
    return AdaRes()

# Define the EfficientNetB7 for classification
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        self.efficientnet = models.efficientnet_b7(pretrained=True)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)

def create_efficientnet_model(num_classes=3):
    return EfficientNetClassifier(num_classes=num_classes)

# Data transformation and loading
def split_dataset_73(dataset_name, size=299, batch_size=32):
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomRotation(30),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale for denoising
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    base_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale for denoising
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    dataset = datasets.ImageFolder(root=dataset_name, transform=base_transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_augmented_dataset = datasets.ImageFolder(root=dataset_name, transform=augment_transform)
    train_augmented_subset = Subset(train_augmented_dataset, train_dataset.indices)
    combined_train_dataset = ConcatDataset([train_dataset, train_augmented_subset])
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_augmented_dataset = datasets.ImageFolder(root=dataset_name, transform=augment_transform)
    combined_dataset = ConcatDataset([dataset, train_augmented_dataset])
    all_train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, all_train_loader

def epoch_forward(train_loader, model, criterion, optimizer, device, task='classification'):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if task == 'denoising':
            loss = criterion(outputs, inputs)  # For denoising, the target is the input image itself
        else:
            labels = labels.long()  # Ensure labels are in long format for classification
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

def save_predictions(predictions, labels, filename):
    with open(filename, 'w') as f:
        for prob, label in zip(predictions, labels):
            f.write(f'{prob.tolist()}, {label}\n')

def cross_val(create_model, num_classes=3, lr=0.001, num_epochs=200, patience=10, train_loader=None, val_loader=None, save_name='', device='cuda'):
    best_val_loss = float('inf')
    model = create_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model = epoch_forward(train_loader, model, criterion, optimizer, device)

        model.eval()
        val_loss = 0.0
        current_preds = []
        current_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                current_preds.extend(outputs.cpu().numpy())
                current_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Best Loss: {best_val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_preds = current_preds
            best_labels = current_labels
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'./pretrained/{save_name}.pth')
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    best_epoch = best_epoch + 1
    save_predictions(best_preds, best_labels, f'./result/{save_name}-{best_epoch}.txt')
    return best_epoch

def train_all(create_model, num_classes=3, lr=0.001, train_loader=None, num_epochs=10, device='cuda', save_name=''):
    model = create_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model = epoch_forward(train_loader, model, criterion, optimizer, device)
    torch.save(model.state_dict(), f'./pretrained/{save_name}.pth')

# Custom Dataset for storing denoised images
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

if __name__ == '__main__':
    num_classes = 3
    lr = 0.0008
    num_epochs = 50  # Reduced epochs for quick debugging; adjust as needed
    batch_size = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ['adares']
    
    # Step 1: Denoising with AdaRes
    for idx, dataset_name in enumerate(['elastic', 'cdfi', '2d']):
        train_loader, val_loader, all_train_loader = split_dataset_73(f'./dataset/{dataset_name}', size=200, batch_size=10)
        adares_model = create_adares_model().to(device)
        adares_criterion = nn.MSELoss()
        adares_optimizer = optim.Adam(adares_model.parameters(), lr=lr)

        # Training AdaRes model
        for epoch in range(num_epochs):
            adares_model = epoch_forward(train_loader, adares_model, adares_criterion, adares_optimizer, device, task='denoising')

        # Save the AdaRes model
        torch.save(adares_model.state_dict(), f'./pretrained/{dataset_name}_adares.pth')
        print(f'{dataset_name}_adares model trained and saved.')

        # Denoise the entire dataset
        denoised_images = []
        denoised_labels = []
        adares_model.eval()
        with torch.no_grad():
            for inputs, labels in all_train_loader:
                inputs = inputs.to(device)
                denoised = adares_model(inputs).cpu()
                denoised_images.append(denoised)
                denoised_labels.extend(labels)
        
        denoised_images = torch.cat(denoised_images)  # Combine all batches into a single tensor

        # Step 2: Classification with EfficientNetB7
        # Convert the denoised images into a DataLoader
        denoised_dataset = CustomDataset(denoised_images, denoised_labels)
        denoised_loader = DataLoader(denoised_dataset, batch_size=batch_size, shuffle=True)

        efficientnet_model = create_efficientnet_model(num_classes=num_classes).to(device)
        efficientnet_criterion = nn.CrossEntropyLoss()
        efficientnet_optimizer = optim.Adam(efficientnet_model.parameters(), lr=lr)

        # Training EfficientNet model
        for epoch in range(num_epochs):
            efficientnet_model = epoch_forward(denoised_loader, efficientnet_model, efficientnet_criterion, efficientnet_optimizer, device, task='classification')

        # Save the EfficientNet model
        torch.save(efficientnet_model.state_dict(), f'./pretrained/{dataset_name}_efficientnet.pth')
        print(f'{dataset_name}_efficientnet model trained and saved.')
