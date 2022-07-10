import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import copy       # helps to make copies instead of references
import time
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

random.seed(1)
food_label = {"0.Rice": 0, "1.Drink": 1, "2.Green Leaf Vegetables": 2, "3.Meat": 3, "4.Noodles": 4}


# class Food_Dataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         """
#         Datasets of Food classification
#         :param data_dir: str, path of food datasets
#         :param transform: torch.transform，data preprocessing
#         """
#         self.label_name = {"0.Rice": 0, "1.Drink": 1, "2.Green Leaf Vegetables": 2, "3.Meat": 3, "4.Noodles": 4}
#         self.data_info = self.get_img_info(data_dir)  # data_info stores the path and the label of every photo,and read samples by index in Dataloader
#         self.transform = transform
#
#     def __getitem__(self, index):
#         path_img, label = self.data_info[index]
#         img = Image.open(path_img).convert('RGB')     # 0~255
#
#         if self.transform is not None:
#             img = self.transform(img)   # perform transformation here,to tensor and so on
#
#         return img, label
#
#     def __len__(self):
#         return len(self.data_info)
#
#     @staticmethod
#     def get_img_info(data_dir):
#         data_info = list()
#         for root, dirs, _ in os.walk(data_dir):
#             # visit each class
#             for sub_dir in dirs:
#                 img_names = os.listdir(os.path.join(root, sub_dir))
#                 img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
#
#                 # visit each photo
#                 for i in range(len(img_names)):
#                     img_name = img_names[i]
#                     path_img = os.path.join(root, sub_dir, img_name)
#                     label = food_label[sub_dir]
#                     data_info.append((path_img, int(label)))
#
#         return data_info
#
# food_dataset = Food_Dataset(r'C:\Users\BiXY\OneDrive - 厦门大学(马来西亚分校)\AIT202\Project\Food_Dataset')

# transform setting
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

data_transform = {
    'train': transforms.Compose([
        transforms.AutoAugment(),
        transforms.RandomResizedCrop(224), # 224 x 224 shrink the picture
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Normalize to [0.0,1.0]
        transforms.Normalize(mean, std) # transform every number in the tensor to be in the range of [-1,1]
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # crop it from the middle
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # crop it from the middle
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


batch_size = 4
data_dir = r'.\Food_Dataset_split'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(class_names)

def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))
    img = std*img + mean
    plt.imshow(img)
    plt.title(title)
    plt.show()

inputs, classes = next(iter(dataloaders['train']))
print(next(iter(dataloaders['train'])))
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

from collections import OrderedDict


# regard one convolutional layer as a basic conv
class BasicConv2d_moveBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(BasicConv2d_moveBN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


# one bottleneck
class Bottleneck_moveBN(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1):
        super(Bottleneck_moveBN, self).__init__()

        self.judge = in_channel == out_channel

        self.bottleneck = nn.Sequential(OrderedDict([
            ('Conv1', BasicConv2d_moveBN(in_channel, mid_channel, 1)),
            ('Relu1', nn.ReLU(True)),
            ('Conv2', BasicConv2d_moveBN(mid_channel, mid_channel, 3, padding=1, stride=stride)),
            ('Relu2', nn.ReLU(True)),
            ('Conv3', BasicConv2d_moveBN(mid_channel, out_channel, 1)), ]
        ))
        self.relu = nn.ReLU(True)
        # here we use linear projection to match dimensions by an 1x1 convolutional layer
        if in_channel != out_channel:
            self.projection = BasicConv2d_moveBN(
                in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.bottleneck(x)
        # perform projection on the residual by an 1x1 convolutional layer if input channel doesn't equal output channel
        if not self.judge:
            self.shortcut = self.projection(x)
            # out += projection of residual
            out += self.shortcut
        # otherwise, add them together directly
        else:
            out += x

        out = self.relu(out)

        return out


# Modified Resnet50:
class ResNet_50_modified(nn.Module):
    def __init__(self, class_num):
        super(ResNet_50_modified, self).__init__()
        self.conv = BasicConv2d_moveBN(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # convolutional block 1
        self.block1 = nn.Sequential(
            Bottleneck_moveBN(64, 64, 256),
            Bottleneck_moveBN(256, 64, 256),
            Bottleneck_moveBN(256, 64, 256),
        )
        # convolutional block 2
        self.block2 = nn.Sequential(
            Bottleneck_moveBN(256, 128, 512, stride=2),
            Bottleneck_moveBN(512, 128, 512),
            Bottleneck_moveBN(512, 128, 512),
            Bottleneck_moveBN(512, 128, 6272),
        )
        self.avgpool = nn.AvgPool2d(4)
        self.classifier = nn.Linear(25088, class_num)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out


def add_graph(writer):
     img = torch.rand([1, 3, 64, 64], dtype=torch.float32)
     model = ResNet_50_modified(5)
     writer.add_graph(model, input_to_model=img)  # 类似于TensorFlow 1.x 中的fed

if __name__ == '__main__':
     writer = SummaryWriter(log_dir=r"C:\Users\BiXY\log", flush_secs=120)
     add_graph(writer)
     writer.close()
