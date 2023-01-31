from autoencoder import Encoder, Decoder, plot_ae_outputs
import torch.nn as nn 
import torch.optim as optim
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.encoder_cnn
        # freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        # add a classifier

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

# load the model
encoder = Encoder(encoded_space_dim=4,fc2_input_dim=128)
encoder.load_state_dict(torch.load('encoder.pth'))
encoder.to(device)

classifier = Classifier(encoder)
params_to_optimize = [
    {'params': classifier.classifier.parameters()}
]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params_to_optimize, lr=0.001, momentum=0.9)

classifier.to(device)

def train_classifier(classifier, train_loader, valid_loader, criterion, optimizer, num_epochs=100):
    train_loss = []
    valid_loss = []
    for epoch in range(num_epochs):
        classifier.train()
        train_loss_epoch = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())
        train_loss.append(np.mean(train_loss_epoch))
        classifier.eval()
        valid_loss_epoch = []
        for i, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            valid_loss_epoch.append(loss.item())
        valid_loss.append(np.mean(valid_loss_epoch))
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Valid Loss: {valid_loss[-1]:.4f}')
    return train_loss, valid_loss
    
num_epochs = 100
train_loss, valid_loss = train_classifier(classifier, train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)

plt.plot(train_loss, label='train loss')
plt.plot(valid_loss, label='valid loss')
plt.legend()
plt.show()

def test_classifier(classifier, test_loader):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

test_classifier(classifier, test_loader)
# save the model
torch.save(classifier.state_dict(), f'classifier_num_epochs{num_epochs}.pth')