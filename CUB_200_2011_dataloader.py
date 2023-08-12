import torch
import torchvision
import config
from torchvision import datasets, transforms
 
data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
def train_data_load():

    root_train = config.ROOT_TRAIN
    train_dataset = torchvision.datasets.ImageFolder(root_train,
                                                     transform=data_transform)
    CLASS = train_dataset.class_to_idx

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.BATCH_SIZE,
                                               shuffle=config.SHUFFLE)
    return CLASS, train_loader
 
def test_data_load():

    root_test = config.ROOT_TEST
    test_dataset = torchvision.datasets.ImageFolder(root_test,
                                                transform=data_transform)
 
    CLASS = test_dataset.class_to_idx

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=config.SHUFFLE)
    return CLASS, test_loader

# train_data_load()