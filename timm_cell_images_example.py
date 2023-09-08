import os
import torch

import matplotlib.pyplot as plt
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import timm
import torch 
import torch.nn.functional as F

from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import transforms

from torch.optim import Adam
from torch.nn import BCELoss

# https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
DATA_FOLDER = os.path.join('datasets', 'cell_images_py')

N_EPOCHS = 5
LR = 0.005

# Alternatively
# test_dataset = ImageFolder(root=DATA_FOLDER, transform=transform)
# print(test_dataset.class_to_idx)

class CellImageDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        
        self.img_labels = []

        uninfected_folder = os.path.join(root, 'Uninfected')
        for foler, dirs, files in os.walk(uninfected_folder):
            for file in files:
                self.img_labels.append([os.path.join(foler, file), 0])

        infected_folder = os.path.join(root, 'Parasitized')
        for foler, dirs, files in os.walk(infected_folder):
            for file in files:
                self.img_labels.append([os.path.join(foler, file), 1])

        self.class_to_idx = {"Uninfected": 0, "Parasitized": 1}
        self.idx_to_class = ["Uninfected", "Parasitized"]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels[idx][0]

        image = Image.open(img_path)
        label = self.img_labels[idx][1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# 0 Define the model
class VisFormer(torch.nn.Module):

    def __init__(self, in_chans=3, out_features=2):
        super(VisFormer, self).__init__()

        self.vit = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True, in_chans=in_chans)

        # Change the classifier
        num_in_features = self.vit.get_classifier().in_features

        self.vit.head = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_in_features, out_features=out_features, bias=False),
            torch.nn.Softmax(dim=1)
        )

        # Alternatively
        # model.reset_classifier(10, 'max')

        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.vit(x)

# 1 Load the dataset
transform = transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])

train_set = CellImageDataset(root=os.path.join(DATA_FOLDER,'train'), transform=transform)
val_set = CellImageDataset(root=os.path.join(DATA_FOLDER, 'val'), transform=transform)
test_set = CellImageDataset(root=os.path.join(DATA_FOLDER, 'test'), transform=transform)

train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
val_loader = DataLoader(val_set, shuffle=False, batch_size=128)
test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# 2 Training loop

model = VisFormer()

# Trainable parameters
trainable_parameters = []

for param in model.parameters():
    if param.requires_grad is True:
        trainable_parameters.append(param)

model.to(device)

optimizer = Adam(trainable_parameters, lr=LR)
bce_loss = BCELoss()

def test_model(model, test_loader, bce_loss):    
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = bce_loss(y_hat, F.one_hot(y, num_classes=2).float())
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        print(f"Test loss: {test_loss:.2f} Test accuracy: {correct / total * 100:.2f}%")

for epoch in range(N_EPOCHS):
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        x, y = x.to(device), y.to(device)

        y_hat = model(x)
        loss = bce_loss(y_hat, F.one_hot(y, num_classes=2).float())

        train_loss += loss.detach().cpu().item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"\nEpoch {epoch + 1}/{N_EPOCHS} Train loss: {train_loss:.2f}")

    # Validate model
    test_model(model, val_loader, bce_loss)

# Test model
test_model(model, test_loader, bce_loss)

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('timm_cell_vit.pt') # Save

# model = torch.jit.load('timm_cell_vit.pt')
# model.eval()
