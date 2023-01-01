import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from balg_utils import race_functions2
import torch
from torchvision.datasets import CIFAR10

class Cifar10Dataset:
    def __init__(self, path):
        ''' Constructor for the cifar10 class. The path variable should point to the folder containing the cifar10 data. '''
        self.path = path
        self.test_data = {}
        self.training_data = {}
        self.classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.load_testdata()
        self.load_traindata()

    def load_testdata(self):
        ''' Loads the test data from the file test_batch. '''
        with open(os.path.join(self.path + 'test_batch'), 'rb') as file:
            self.test_data = pickle.load(file, encoding='bytes')
    def load_traindata(self):
        ''' Loads the training data from the file data_batch_1. '''
        with open(os.path.join(self.path + 'data_batch_1'), 'rb') as file:
            self.training_data = pickle.load(file, encoding='bytes')
        with open(os.path.join(self.path + 'data_batch_2'), 'rb') as file:
            #append data from data_batch_2 to training_data
            data = pickle.load(file, encoding='bytes')
            self.training_data[b'data'] = np.append(self.training_data[b'data'], data[b'data'], axis=0)
            self.training_data[b'labels'] = np.append(self.training_data[b'labels'], data[b'labels'], axis=0)
            self.training_data[b'filenames'] = np.append(self.training_data[b'filenames'], data[b'filenames'], axis=0)
        with open(os.path.join(self.path + 'data_batch_3'), 'rb') as file:
            #append data from data_batch_3 to training_data
            data = pickle.load(file, encoding='bytes')
            self.training_data[b'data'] = np.append(self.training_data[b'data'], data[b'data'], axis=0)
            self.training_data[b'labels'] = np.append(self.training_data[b'labels'], data[b'labels'], axis=0)
            self.training_data[b'filenames'] = np.append(self.training_data[b'filenames'], data[b'filenames'], axis=0)
        with open(os.path.join(self.path + 'data_batch_4'), 'rb') as file:
            #append data from data_batch_4 to training_data
            data = pickle.load(file, encoding='bytes')
            self.training_data[b'data'] = np.append(self.training_data[b'data'], data[b'data'], axis=0)
            self.training_data[b'labels'] = np.append(self.training_data[b'labels'], data[b'labels'], axis=0)
            self.training_data[b'filenames'] = np.append(self.training_data[b'filenames'], data[b'filenames'], axis=0)
        with open(os.path.join(self.path + 'data_batch_5'), 'rb') as file:
            #append data from data_batch_5 to training_data
            data = pickle.load(file, encoding='bytes')
            self.training_data[b'data'] = np.append(self.training_data[b'data'], data[b'data'], axis=0)
            self.training_data[b'labels'] = np.append(self.training_data[b'labels'], data[b'labels'], axis=0)
            self.training_data[b'filenames'] = np.append(self.training_data[b'filenames'], data[b'filenames'], axis=0)
    
    def get_image_with_label(self, index, reshape=True):
        ''' Returns the image and the label for the given index. '''
        if reshape:
            return self.training_data[b'data'][index].reshape(3, 32, 32).transpose(1, 2, 0), self.training_data[b'labels'][index]
        return self.training_data[b'data'][index], self.training_data[b'labels'][index]

    def get_classname(self, index):
        ''' Returns the classname at index. '''
        return self.classnames[index]


class KNNClassifier:
    def __init__(self, training_data, training_labels):
        ''' Constructor for the KNNClassifier class. '''
        #normalize the training data 
        training_data = self.normalize(self, training_data)
        self.training_data = training_data
        self.training_labels = training_labels
        
    def classify(self, test_data, k):
        ''' Classifies the test data using the k-nearest neighbor algorithm. '''
        #compute the distance between the test data and the training data (TODO)
        distances = self.compute_l2distance(test_data)
        #find k nearest neighbors (TODO)
        neighbors = self.find_k_nearest_neighbors(distances, k)
        #compute the class label for each test data point (TODO)
        labels = {}
        #compute the accuracy of the classification (TODO)
        accuracy = 0
        return labels, accuracy

    def compute_l2distance(self, test_data):
        ''' Computes the distance between the test data and the training data. '''
        #compute the distance between the test data and the training data (TODO)
        distances = {}
        return distances

    def find_k_nearest_neighbors(self, distances, k):
        ''' Finds the k nearest neighbors for each test data point. '''
        #find k nearest neighbors (TODO)
        neighbors = {}
        return neighbors

    def compute_class_label(self, neighbors):
        ''' Computes the class label for each test data point. '''
        #compute the class label for each test data point (TODO)
        labels = {}
        return labels

    def normalize(self, data):
        ''' Normalizes the data using the mean and standard deviation of the training data. '''
        #normalize the data 
        dev = np.std(data, axis=0)
        mean = np.mean(data, axis=0)
        data = (data - mean) / dev
        return data



# Create an instance of the cifar10 class
dataset = Cifar10Dataset('data/cifar10/')

# load 5 randowm images and display them along with their labels
for i in range(5):
    image_index = np.random.randint(0, 10000)
    image, label = dataset.get_image_with_label(image_index, reshape=True)
    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    plt.title('Label: ' + dataset.get_classname(label))
plt.show()