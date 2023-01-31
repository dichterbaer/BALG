import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Pretrained AlexNet works on 224 x 224 x 3 images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Loading training and test data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)


#Show some random images for illustration
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# #Get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# #Show images
# imshow(torchvision.utils.make_grid(images))


#Use pretrained AlexNet
alexnet = models.alexnet(pretrained=True)
alexnet.to(device)
print(alexnet)

#Freeze parameters
for p in alexnet.features.parameters():
            p.requires_grad = False


#modify sizes fully connected (look up torch.nn.Linear)
alexnet.classifier[4] = nn.Linear(in_features=4096, out_features=1024, bias=True)
alexnet.classifier[6] = nn.Linear(in_features=1024, out_features=10, bias=True)
print(alexnet)

alexnet.to(device)

#Loss
criterion = nn.CrossEntropyLoss()


PATH = f'cifar_alexnet_transfer_lr_0.001_momentum_0.9_last_loss_0.8667344450950623.pth'
alexnet.load_state_dict(torch.load(PATH))
#Optimizer(SGD)
#optimizer = (look up torch.optim.SGD)
# lr = 0.001
# momentum = 0.9
# optimizer = optim.SGD(alexnet.parameters(), lr=lr, momentum=momentum)


# for epoch in range(10):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         output = alexnet(inputs)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

#     #print('Finished Epoch Training of AlexNet')

#     #Testing Accuracy
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs = alexnet(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#         100 * correct / total))

#Testing Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
#Testing classification accuracy for individual classes.
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))   

# print('Finished Training of AlexNet')
# torch.save(alexnet.state_dict(), PATH)
