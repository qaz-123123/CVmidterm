# -*- coding: utf-8 -*-
"""
Created on Sun May 26 23:54:58 2024

@author: zheng'ling'fei
"""

import torch
from torchvision import models, transforms
# import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt


# torch.manual_seed(12345)

# alexnet = models.alexnet(pretrained=True)
# print(alexnet)

# from torch.hub import set_dir

# set_dir('D://Python//Python310')
# print(torch.hub.get_dir())
#
# alexnet = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
# print(alexnet)

# alexnet = torch.load('D://Python//Python310//checkpoints//alexnet-owt-7be5be79.pth')
# print(alexnet)

# alexnet = models.alexnet(pretrained=True)

# newLinear = torch.nn.Linear(in_features=alexnet.classifier[6].in_features, out_features=200, bias=True)
# alexnet.classifier[6] = newLinear

resnet18 = models.resnet18(pretrained=True)
# print(resnet18)
newLinear = torch.nn.Linear(in_features=resnet18.fc.in_features, out_features=200, bias=True)
resnet18.fc = newLinear

# resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
# print(resnet18)

# train_dataset = datasets.ImageFolder(root='CUB_200_2011', transform=train_transform)
# test_dataset = datasets.ImageFolder(root='CUB_200_2011', transform=test_transform)
#
#
# train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_data = DataLoader(test_dataset, batch_size=32, shuffle=True)

# optimizer = torch.optim.SGD(alexnet.parameters(), lr=pow(10, -5))
# loss_fn = torch.nn.CrossEntropyLoss()


class CustomDataset(Dataset):
    def __init__(self, data_dir, split_file, class_file, image_class_file, images_file, train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = []

        with open(class_file, 'r') as f:
            class_lines = f.readlines()
            self.class_dict = {int(line.strip().split()[0]): line.strip().split()[1] for line in class_lines}

        with open(image_class_file, 'r') as f:
            image_class_lines = f.readlines()
            self.image_class_dict = {int(line.strip().split()[0]): int(line.strip().split()[1]) for line in image_class_lines}

        with open(images_file, 'r') as f:
            images_lines = f.readlines()
            self.images_dict = {int(line.strip().split()[0]): line.strip().split()[1] for line in images_lines}

        with open(split_file, 'r') as f:
            split_lines = f.readlines()

        for line in split_lines:
            image_id, is_train = line.strip().split()
            image_id = int(image_id)
            is_train = int(is_train)
            if (train and is_train == 1) or (not train and is_train == 0):
                self.file_list.append([image_id, self.image_class_dict[image_id]-1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_id, class_id = self.file_list[idx]
        img_filename = self.images_dict[img_id]
        img_name = os.path.join(self.data_dir, 'images', img_filename)
        image = Image.open(img_name).convert('RGB').resize((224, 224))

        if self.transform:
            image = self.transform(image)

        return image, class_id


data_dir = r'CUB_200_2011'
images_file = r'CUB_200_2011//images.txt'
split_file = r'CUB_200_2011//train_test_split.txt'
class_file = r'CUB_200_2011//classes.txt'
image_class_file = r'CUB_200_2011//image_class_labels.txt'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataset = CustomDataset(data_dir, split_file, class_file, image_class_file, images_file, train=True, transform=train_transform)
test_dataset = CustomDataset(data_dir, split_file, class_file, image_class_file, images_file, train=False, transform=test_transform)

train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=32, shuffle=True)


def train_model(device, lr):
    # optimizer = torch.optim.SGD(alexnet.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, resnet18.parameters()), lr)
    criterion = torch.nn.CrossEntropyLoss()

    resnet18.train()
    running_loss = 0.0
    for images, labels in train_data:
        # print(labels)
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        # print(images.shape)
        outputs = resnet18(images)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_data)
    print(f'Epoch {epoch} Loss: {epoch_loss:.4f}')
    return epoch_loss


def test_model(device):
    resnet18.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            outputs = resnet18(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy


epoch_num = 100
lr = pow(10, -3)
decay = 0.9

epochs_loss_list = []
accuracy_list = []

# optimizer = torch.optim.Adam(alexnet.parameters(), lr)

epochs_frozen = 30
epochs_unfrozen = 200
epoch_num = epochs_frozen + epochs_unfrozen

for param in resnet18.parameters():
    param.requires_grad = False
    
for param in resnet18.fc.parameters():
    param.requires_grad = True
    

# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet18.parameters()), lr)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, resnet18.parameters()), lr)

for epoch in range(epochs_frozen):
    lr = lr * decay
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, resnet18.parameters()), lr)
    
    epoch_loss = train_model('cpu', lr)
    epochs_loss_list.append(epoch_loss)
    epoch_accuracy = test_model('cpu')
    accuracy_list.append(epoch_accuracy)
    
for param in resnet18.parameters():
    param.requires_grad = True
    
# base_params = filter(lambda p: id(p) not in resnet18.fc.parameters(), resnet18.parameters())
base_params = [p for p in resnet18.parameters() if id(p) not in [id(param) for param in resnet18.fc.parameters()]]

for epoch in range(epochs_frozen, epoch_num):
    lr = lr * decay if not epoch%10 else lr
    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': resnet18.fc.parameters(), 'lr': lr}], lr=pow(10, -3))
    epoch_loss = train_model('cpu', lr)
    epochs_loss_list.append(epoch_loss)
    epoch_accuracy = test_model('cpu')
    accuracy_list.append(epoch_accuracy)

fig, ax1 = plt.subplots()
ax1.plot(range(epoch_num), epochs_loss_list, color='blue', label='training loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')

ax2 = ax1.twinx()
ax2.plot(range(epoch_num), accuracy_list, color='red', label='training accuracy')
ax2.set_ylabel('Accuracy')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc='upper right')

plt.title('Loss and Accuracy')
plt.savefig('figure//resnet18//epochfro30&epochunfro200&SGD&stratifiedlr&decay09.png', dpi=300)
plt.show()

torch.save(resnet18, 'save_models//trained_resnet18//epochfro30&epochunfro200&SGD&stratifiedlr&decay09.pth')