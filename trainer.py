import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from sklearn.model_selection import KFold

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def split_dataset_73(dataset_name, size=299, batch_size=32):

    augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.RandomRotation(30),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    base_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    return train_loader,val_loader,all_train_loader



def split_dataset_kfolds(dataset_name, size=299, batch_size=32):
    ls = []
    augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.RandomRotation(30),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    base_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=dataset_name, transform=base_transform)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)
        
        train_augmented_dataset = datasets.ImageFolder(root=dataset_name, transform=augment_transform)
        train_augmented_subset = Subset(train_augmented_dataset, train_index)
        combined_train_subset = ConcatDataset([train_subset, train_augmented_subset])
        
        train_loader = DataLoader(combined_train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        ls.append((train_loader, val_loader))

    train_augmented_dataset = datasets.ImageFolder(root=dataset_name, transform=augment_transform)
    combined_dataset = ConcatDataset([dataset, train_augmented_dataset])
    all_train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return ls, all_train_loader


def epoch_forward(train_loader, model, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  
        optimizer.zero_grad()
        outputs = model(inputs)
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


def cross_val_kfold(create_model, num_classes=3, lr=0.001,num_epochs=20, kfolds=None, save_name='', device='cuda'):
    best_epochs = 0
    for fold, (train_loader, val_loader) in enumerate(kfolds):
        best_epochs +=  cross_val(create_model=create_model, num_classes=num_classes, lr=lr, num_epochs=num_epochs, train_loader=train_loader,val_loader=val_loader, save_name=f'{save_name}-{fold}', device=device)
    return np.ceil(best_epochs/5)



