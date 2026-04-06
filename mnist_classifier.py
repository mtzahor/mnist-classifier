import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define transformations for the MNIST data (normalize only)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and prepare the datasets and loaders
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define a simple feedforward neural network for MNIST with Dropout
class LinearMNISTClassifier(nn.Module):
    def __init__(self):
        super(LinearMNISTClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.model(x)

# Define a CNN Model for MNIST
class CNNMNISTClassifier(nn.Module):
    def __init__(self):
        super(CNNMNISTClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to train a model for one epoch
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    return running_loss / total, 100. * correct / total

# Function to test a model
def test_epoch(model, loader, criterion, device, return_preds=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            if return_preds:
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
    avg_loss = test_loss / total
    acc = 100. * correct / total
    if return_preds:
        return avg_loss, acc, np.array(all_targets), np.array(all_preds)
    return avg_loss, acc

def run_training(model, train_loader, test_loader, optimizer, criterion, device, epochs=5):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Epoch {epoch}:")
        print(f"  Train loss: {train_loss:.4f}, accuracy: {train_acc:.2f}%")
        print(f"  Test  loss: {test_loss:.4f}, accuracy: {test_acc:.2f}%\n")
    return train_losses, test_losses, train_accs, test_accs

# Instantiate models, criterion, and optimizers
linear_model = LinearMNISTClassifier().to(device)
cnn_model = CNNMNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
linear_optimizer = optim.Adam(linear_model.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

epochs = 5

# Train Linear Model
print("Training Linear (Dense) Model")
lin_train_losses, lin_test_losses, lin_train_accs, lin_test_accs = run_training(
    linear_model, train_loader, test_loader, linear_optimizer, criterion, device, epochs=epochs
)
# Train CNN Model
print("Training CNN Model")
cnn_train_losses, cnn_test_losses, cnn_train_accs, cnn_test_accs = run_training(
    cnn_model, train_loader, test_loader, cnn_optimizer, criterion, device, epochs=epochs
)

# Final Evaluations for Confusion Matrix
_, _, lin_y_true, lin_y_pred = test_epoch(linear_model, test_loader, criterion, device, return_preds=True)
_, _, cnn_y_true, cnn_y_pred = test_epoch(cnn_model, test_loader, criterion, device, return_preds=True)

# Visualization: Training & Test Curves
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
epochs_range = np.arange(1, epochs + 1)

# Accuracy curves
axs[0, 0].plot(epochs_range, lin_train_accs, label='Linear Train Acc', marker='o')
axs[0, 0].plot(epochs_range, lin_test_accs, label='Linear Test Acc', marker='o')
axs[0, 0].plot(epochs_range, cnn_train_accs, label='CNN Train Acc', marker='s')
axs[0, 0].plot(epochs_range, cnn_test_accs, label='CNN Test Acc', marker='s')
axs[0, 0].set_title('Accuracy over Epochs')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy (%)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Loss curves
axs[0, 1].plot(epochs_range, lin_train_losses, label='Linear Train Loss', marker='o')
axs[0, 1].plot(epochs_range, lin_test_losses, label='Linear Test Loss', marker='o')
axs[0, 1].plot(epochs_range, cnn_train_losses, label='CNN Train Loss', marker='s')
axs[0, 1].plot(epochs_range, cnn_test_losses, label='CNN Test Loss', marker='s')
axs[0, 1].set_title('Loss over Epochs')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Confusion Matrix for Linear Model
cm_lin = confusion_matrix(lin_y_true, lin_y_pred)
disp_lin = ConfusionMatrixDisplay(cm_lin, display_labels=[str(i) for i in range(10)])
disp_lin.plot(ax=axs[1, 0], cmap=plt.cm.Blues)
axs[1, 0].set_title("Confusion Matrix - Linear")

# Confusion Matrix for CNN Model
cm_cnn = confusion_matrix(cnn_y_true, cnn_y_pred)
disp_cnn = ConfusionMatrixDisplay(cm_cnn, display_labels=[str(i) for i in range(10)])
disp_cnn.plot(ax=axs[1, 1], cmap=plt.cm.Blues)
axs[1, 1].set_title("Confusion Matrix - CNN")

plt.tight_layout()
plt.show()