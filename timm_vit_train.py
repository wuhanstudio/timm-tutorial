import timm
import torch 
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import transforms

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Model selection
# https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv

N_EPOCHS = 10
LR = 0.005

# 0 Define the model
class VisFormer(torch.nn.Module):

    def __init__(self, in_chans=1, out_features=10):
        super(VisFormer, self).__init__()

        self.vit = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True, in_chans=in_chans)

        # Change the classifier
        num_in_features = self.vit.get_classifier().in_features

        self.vit.head = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(num_in_features),
            # torch.nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(512),
            # torch.nn.Dropout(0.4),
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
    transforms.Resize(size=224),
    transforms.ToTensor()
])

train_set = MNIST(root='./datasets', train=True, download=True, transform=transform)
test_set = MNIST(root='./datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
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
ce_loss = CrossEntropyLoss()

def test_model(model, test_loader, ce_loss):    
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = ce_loss(y_hat, y)
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
        loss = ce_loss(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"\nEpoch {epoch + 1}/{N_EPOCHS} Train loss: {train_loss:.2f}")

    # Test model
    test_model(model, test_loader, ce_loss)

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('timm_vit.pt') # Save

# model = torch.jit.load('timm_vit.pt')
# model.eval()
