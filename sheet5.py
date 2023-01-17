import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from balg_utils import race_functions2
from skimage.feature import hog

class Cifar10Dataset:
    def __init__(self, path, num_train_samples=50000):
        ''' Constructor for the cifar10 class. The path variable should point to the folder containing the cifar10 data. '''
        self.path = path
        self.test_data = {}
        self.training_data = {}
        self.classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.load_testdata()
        self.load_traindata(num_train_samples)

    def load_testdata(self):
        ''' Loads the test data from the file test_batch. '''
        with open(os.path.join(self.path + 'test_batch'), 'rb') as file:
            self.test_data = pickle.load(file, encoding='bytes')
    def load_traindata(self, num_train_samples):
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
        #only keep the first num_train_samples samples
        self.training_data[b'data'] = self.training_data[b'data'][:num_train_samples]
    
    def get_train_image_with_label(self, index, reshape=True):
        ''' Returns the image and the label for the given index. '''
        if reshape:
            return self.training_data[b'data'][index].reshape(3, 32, 32).transpose(1, 2, 0), self.training_data[b'labels'][index]
        return self.training_data[b'data'][index], self.training_data[b'labels'][index]

    def get_test_image_with_label(self, index, reshape=True):
        ''' Returns the image and the label for the given index. '''
        if reshape:
            return self.test_data[b'data'][index].reshape(3, 32, 32).transpose(1, 2, 0), self.test_data[b'labels'][index]
        return self.test_data[b'data'][index], self.test_data[b'labels'][index]

    def get_classname(self, index):
        ''' Returns the classname at index. '''
        return self.classnames[index]


class KNNClassifier:
    def __init__(self, dataset):
        ''' Constructor for the KNNClassifier class. '''
        #normalize the training data 
        training_data = self.normalize(dataset.training_data[b'data'])
        self.x_train = training_data
        self.y_train = dataset.training_data[b'labels']
        test_data = self.normalize(dataset.test_data[b'data'])
        self.x_test = test_data
        self.y_test = np.array(dataset.test_data[b'labels'])

    def create_test_dataset(self):
        ''' Creates a test dataset with small shapes for debugging. '''
        training_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        training_labels =np.array([1, 2, 3])
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        test_labels = np.array([1, 2])
        self.x_train = training_data
        self.y_train = training_labels
        self.x_test = test_data
        self.y_test = test_labels
        self.x_train_val = []
        self.y_train_val = []
        self.x_test_val = []
        self.y_test_val = []

    def use_hog_features(self, hog_train, hog_test):
        ''' Uses the HOG features for the training and test data. '''
        self.x_train = self.normalize(hog_train)
        self.x_test = self.normalize(hog_test)

    def cross_validation(self, num_samples, num_folds = 5, k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100], verbose = True):
        ''' Performs a cross validation on num_samples samples of the training data. '''
        #shuffle the training data
        indices = np.arange(self.x_train.shape[0])
        np.random.shuffle(indices)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]
        #get the first num_samples samples
        x_train = self.x_train[:num_samples]
        y_train = self.y_train[:num_samples]
        samples_per_fold = int(num_samples / num_folds)
        accs = []
        for k in k_choices:
            acc_fold = []
            for i in range(0, num_folds):
                self.x_train_val = x_train[:i * samples_per_fold]
                self.y_train_val = y_train[:i * samples_per_fold]
                self.x_train_val = np.append(self.x_train_val, x_train[(i + 1) * samples_per_fold:], axis=0)
                self.y_train_val = np.append(self.y_train_val, y_train[(i + 1) * samples_per_fold:], axis=0)
                self.x_test_val = x_train[i * samples_per_fold:(i + 1) * samples_per_fold]
                self.y_test_val = y_train[i * samples_per_fold:(i + 1) * samples_per_fold]
                acc = self.check_accuracy(test_data_indices = None, k=k, cross_val = True)
                acc_fold.append(acc)
                print()
            #compute the mean accuracy for the current k
            accs.append(np.mean(acc_fold))
            #print the mean accuracy for the current k
            if verbose:
                print('The mean accuracy for k = ' + str(k) + ' is: ' + str(accs[-1]))

        #plot the accuracies for the different k values
        plt.plot(k_choices, accs)
        plt.xlabel('k')
        plt.ylabel('Cross-validation accuracy')
        plt.title('Cross-validation on k, num_samples = ' + str(num_samples) + ', num_folds = ' + str(num_folds))
        plt.show()
        #print best k
        if verbose:
            print('The best k is: ', k_choices[np.argmax(accs)])
        #return the best k
        return k_choices[np.argmax(accs)]

    def predict(self, test_data_indices, k=1, cross_val = False, verbose=False):
        ''' predict the test data using the k-nearest neighbor algorithm. '''
        if(cross_val):
            x_test = self.x_test_val
        else:
            # Get the test data
            x_test = self.x_test[test_data_indices]
        #check if the test data is a 1D array
        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        # Init output shape
        y_test_pred = np.zeros(x_test.shape[0], dtype=np.int64)
        #compute the distance between the test data and the training data
        #distances = self.compute_l2distance(x_test, verbose)
        distances = self.compute_l2_distance_identity(x_test, cross_val=cross_val, verbose = verbose)
        
        for i in range(0, distances.shape[0]):
            # Find index of k lowest values
            x = np.argpartition(distances[i, :], k)[:k]
            # Index the labels according to x
            if(cross_val):
                k_lowest_labels = self.y_train_val[x]
            else:
                k_lowest_labels = self.y_train[x]
            # y_test_pred[i] = the most frequent occuring index
            y_test_pred[i] = np.bincount(k_lowest_labels).argmax()
        return y_test_pred

    def check_accuracy(self, test_data_indices, k=1, cross_val = False, verbose=True):
        ''' Check the accuracy of the classification. '''
        if(cross_val):
            test_data_indices = np.arange(self.x_test_val.shape[0])
        else:
            if(test_data_indices == None):
                test_data_indices = np.arange(self.x_test.shape[0])
        y_test_pred = self.predict(test_data_indices, k, cross_val=cross_val, verbose=verbose)
        if(cross_val):
            y_test = self.y_test_val
        else:
            y_test = self.y_test[test_data_indices]
        num_samples = len(test_data_indices)
        num_correct = np.sum(y_test_pred == y_test)
        accuracy = float(num_correct) / num_samples
        msg = (f'Got {num_correct} / {num_samples} correct; ' f'accuracy is {(accuracy*100):.2f}%, k = {k}')
        print(msg)
        return accuracy

    def compute_l2distance(self, test_data, verbose=False):
        ''' Computes the distance between the test data and the training data. '''
        #compute the distance between the test data and the training data 
        num_test = test_data.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(0, num_test):
            for j in range(0, num_train):
                dists[i, j] = np.sqrt(np.sum(np.square(self.x_train[j] - test_data[i])))
            if verbose:
                print('Progress predicting test data: {}/{}'.format(i+1, num_test))
        return dists


    def compute_l2_distance_identity(self, test_data, cross_val = False, verbose=False):
        ''' Computes the distance between the test data and the training data with the identity. '''
        num_test = test_data.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train))
        if(cross_val):
            train_sq = np.square(self.x_train_val)
        else:
            train_sq = np.square(self.x_train)
        test_sq = np.square(test_data)
        test_sum_sq = np.sum(test_sq, axis=1)
        train_sum_sq = np.sum(train_sq, axis=1)
        if(cross_val):
            mul = np.matmul(self.x_train_val, np.transpose(test_data))
        else:
            mul = np.matmul(self.x_train, np.transpose(test_data))
        dists = np.sqrt(train_sum_sq.reshape(-1, 1) + test_sum_sq.reshape(1, -1) - 2 * mul).transpose()
        return dists

    def normalize(self, data):
        ''' Normalizes the data using the mean and standard deviation of the training data. '''
        #normalize the data 
        dev = np.std(data, axis=0)
        mean = np.mean(data, axis=0)
        data = (data - mean) / dev
        return data


class SoftmaxClassifier():
    def __init__(self, dataset):
        self.dataset = dataset
        self.x_train = self.normalize(dataset.training_data[b'data'])
        self.y_train = dataset.training_data[b'labels']
        self.x_test = self.normalize(dataset.test_data[b'data'])
        self.y_test = dataset.test_data[b'labels']
        self.W = None

    def use_hog_features(self, hog_train, hog_test):
        ''' Uses the HOG features for the training and test data. '''
        # self.x_train = self.normalize(hog_train)
        # self.x_test = self.normalize(hog_test)
        self.x_train = self.normalize(np.array(hog_train))
        self.x_test = self.normalize(np.array(hog_test))



    def train(self, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        ''' Train the classifier using stochastic gradient descent. '''
        num_train, dim = self.x_train.shape
        num_classes = np.max(self.y_train) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Sample batch_size elements from the training data and their corresponding labels to use in this round of gradient descent.
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = self.x_train[indices]
            y_batch = self.y_train[indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W += -learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                
        return loss_history

    def loss(self, X_batch, y_batch, reg):
        ''' Compute the loss function and its derivative. '''
        # initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)

        # compute the loss and the gradient
        num_train = X_batch.shape[0]
        scores = X_batch.dot(self.W)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_train), y_batch])
        loss = np.sum(correct_logprobs) / num_train
        loss += 0.5 * reg * np.sum(self.W * self.W)

        dscores = probs
        dscores[range(num_train), y_batch] -= 1
        dscores /= num_train
        dW = np.dot(X_batch.T, dscores)
        dW += reg * self.W

        return loss, dW
        
    def check_accuracy(self, test_data_indices=None, verbose=False):
        ''' Check the accuracy of the classifier on the test data. '''
        if test_data_indices is None:
            test_data_indices = range(self.x_test.shape[0])
        num_correct = 0
        num_test = len(test_data_indices)
        for i, idx in enumerate(test_data_indices):
            scores = self.x_test[idx].dot(self.W)
            y_pred = np.argmax(scores)
            if y_pred == self.y_test[idx]:
                num_correct += 1
            if verbose and i % 100 == 0:
                print('Progress predicting test data: {}/{}'.format(i+1, num_test))
        acc = float(num_correct) / num_test
        msg = (f'Got {num_correct} / {num_test} correct; ' f'accuracy is {(acc*100):.2f}%')
        print(msg)
        return acc

    def normalize(self, data):
        ''' Normalizes the data using the mean and standard deviation of the training data. '''
        #normalize the data 
        dev = np.std(data, axis=0)
        mean = np.mean(data, axis=0)
        data = (data - mean) / dev
        return data


# Create an instance of the cifar10 class
dataset = Cifar10Dataset('data/cifar10/', 50000)

problem = 4
if(problem == 1):
    knn = KNNClassifier(dataset)
    #k_choices=[1, 5, 10, 20, 50, 100]
    k_choices=[1, 3, 5, 7, 10, 12]
    best_k = knn.cross_validation(num_samples = 10000, num_folds=5, k_choices=k_choices)
    test_data_indices = None
    acc = knn.check_accuracy(test_data_indices, k=best_k, cross_val = False, verbose=True)
    #predict 5 random images
    for i in range(5):
        image_index = np.random.randint(0, 10000)
        prediction = knn.predict(image_index, k=best_k)
        predicted_class_name = dataset.classnames[int(prediction)]
        image, label = dataset.get_image_with_label(image_index, reshape=True)
        plt.subplot(1, 5, i+1)
        plt.imshow(image)
        plt.title('Label: {}, Predicted Label: {}'.format(dataset.classnames[label], predicted_class_name))
    plt.show()

# #labels= knn.predict(test_data_indices, 1)
# acc = knn.check_accuracy(test_data_indices, k=3, verbose=True)
if(problem == 2):
        # Train a linear classifier
    dataset = Cifar10Dataset('data/cifar10/', 50000)
    softmax = SoftmaxClassifier(dataset)
    softmax.train(learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=True)
    acc = softmax.check_accuracy(verbose=True)
    #predict 5 random images
    # for i in range(5):
    #     image_index = np.random.randint(0, 10000)
    #     image, label = softmax.dataset.get_image_with_label(image_index, reshape=True)
    #     label = softmax.predict(image_index)
    #     predicted_class_name = softmax.dataset.classnames[label]
    #     actual_class_name = softmax.dataset.classnames[softmax.y_test[image_index]]
    #     plt.subplot(1, 5, i+1)
    #     plt.imshow(image)
    #     plt.title('Label: {}, Predicted label: {},'.format(actual_class_name, predicted_class_name))
    #     #print('Predicted label: {},'.format(predicted_class_name))
    # plt.show()


if(problem == 3):
    # HOG features
    dataset = Cifar10Dataset('data/cifar10/', 50000)
    hog_train = []
    hog_test = []
    for i in range(50000):
        image_train, label_train = dataset.get_train_image_with_label(i, reshape=True)
        fd_train = hog(image_train, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=False, multichannel=True)
        hog_train.append(fd_train)
        if(i < 10000):
            image_test, label_test = dataset.get_test_image_with_label(i, reshape=True)
            fd_test = hog(image_test, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=False, multichannel=True)
            hog_test.append(fd_test)


    knn = KNNClassifier(dataset)
    knn.use_hog_features(hog_train, hog_test)
    k_choices=[1, 3, 5, 7, 10, 12, 15, 20]
    best_k = knn.cross_validation(num_samples = 10000, num_folds=5, k_choices=k_choices)
    test_data_indices = None
    acc = knn.check_accuracy(test_data_indices, k=best_k, cross_val = False, verbose=True)

    


if(problem == 4):
    # HOG features
    dataset = Cifar10Dataset('data/cifar10/', 50000)
    hog_train = []
    hog_test = []
    pixel_per_cell = (8, 8)
    cells_per_block = (4, 4)
    for i in range(50000):
        image_train, label_train = dataset.get_train_image_with_label(i, reshape=True)
        fd_train = hog(image_train, orientations=9, pixels_per_cell=pixel_per_cell, cells_per_block=cells_per_block, visualize=False, multichannel=True)
        hog_train.append(fd_train)
        if(i < 10000):
            image_test, label_test = dataset.get_test_image_with_label(i, reshape=True)
            fd_test = hog(image_test, orientations=9, pixels_per_cell=pixel_per_cell, cells_per_block=cells_per_block, visualize=False, multichannel=True)
            hog_test.append(fd_test)

    softmax = SoftmaxClassifier(dataset)
    softmax.use_hog_features(hog_train, hog_test)
    softmax.train(learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=True)
    acc = softmax.check_accuracy(verbose=True)