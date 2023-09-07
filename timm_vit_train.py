import timm
import torch 
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import transforms

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

N_EPOCHS = 10
LR = 0.005

# 1 Load the dataset
transform = transforms = transforms.Compose([
    transforms.Resize(size=224),
    transforms.ToTensor()
])

train_set = MNIST(root='./datasets', train=True, download=True, transform=transform)
test_set = MNIST(root='./datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

# 2 Initialize the pretrained model
model = timm.create_model('visformer_tiny.in1k', pretrained=True, in_chans=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# Freeze parameters
trainable_parameters = []

for param in model.parameters():
    param.requires_grad = False

# 3 Change the classifier
num_in_features = model.get_classifier().in_features

model.head = torch.nn.Sequential(
    # torch.nn.BatchNorm1d(num_in_features),
    # torch.nn.Linear(in_features=num_in_features, out_features=512, bias=False),
    # torch.nn.ReLU(),
    # torch.nn.BatchNorm1d(512),
    # torch.nn.Dropout(0.4),
    torch.nn.Linear(in_features=num_in_features, out_features=10, bias=False),
    torch.nn.Softmax(dim=1)
)

trainable_parameters = model.head.parameters()

# Alternatively
# model.reset_classifier(10, 'max')

# for param in model.fc.parameters():
#     param.requires_grad = True
#     trainable_parameters.append(param)

# 4 Training loop

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


optimizer = Adam(trainable_parameters, lr=LR)
ce_loss = CrossEntropyLoss()

model.to(device)

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
