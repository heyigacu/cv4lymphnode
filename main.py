import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms, models
from timm import create_model
from trainer import cross_val,cross_val_kfold,train_all,split_dataset_73,split_dataset_kfolds, split_dataset_73_adares
from adares import AdaRes

def create_resnet_model(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_efficientnet_model(num_classes):
    model = EfficientNet.from_name('efficientnet-b7')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def create_convnext_model(num_classes):
    model = create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
    return model

def create_vit_model(num_classes):
    model = create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    return model

def create_adares_model(num_classes):
    return AdaRes()



if __name__ == '__main__':
    num_classes = 3
    lr = 0.0008
    num_epochs = 200
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ['ResNet18', 'efficientnet_b7', 'ConvNeXt', 'ViT']
    model_names = ['adares']
    for idx,dataset_name in enumerate(['elastic','cdfi', '2d']):
        for idx,create in enumerate([create_resnet_model,create_efficientnet_model,create_convnext_model,create_vit_model]):
            save_name='{}-{}'.format(dataset_name, model_names[idx])
            # if model_names[idx] == 'adares':
            #     train_loader,val_loader,all_train_loader = split_dataset_73_adares(f'./dataset/{dataset_name}', size=200, batch_size=10)
            if model_names[idx] != 'efficientnet_b7':
                train_loader,val_loader,all_train_loader = split_dataset_73(f'./dataset/{dataset_name}', size=224, batch_size=16)
            else:
                train_loader,val_loader,all_train_loader = split_dataset_73(f'./dataset/{dataset_name}', size=299, batch_size=16)
            best_epoch = cross_val(create, num_classes=num_classes, lr=lr, num_epochs=num_epochs, train_loader=train_loader,val_loader=val_loader, save_name=save_name, device=device)
            print(save_name,'best-epoch:',best_epoch)
            train_all(create, num_classes=num_classes, lr=lr, train_loader=all_train_loader, num_epochs=best_epoch, device=device, save_name='{}-{}'.format(save_name,best_epoch))
            
         
         
