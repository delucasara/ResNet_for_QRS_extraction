from numpy import vstack
from torchsummary import summary
import numpy as np
from pandas import read_csv
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import SGD, Adam, Adagrad
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from matplotlib import pyplot as plt
from ResNet1d import *
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import math
import time

import h5py
import helpers
from pathlib import Path

 
# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path, normalize = False):
        global min_y
        global max_y
        # load the csv file as a dataframe
        df = read_csv(path)
        df = df.transpose()
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        # print(self.X)
        self.y = df.values[:, -1]      
        #print("Dataset length: ", self.X.shape[0])
        
        # ensure input data is floats
        self.X = self.X.astype(np.float)
        self.y = self.y.astype(np.float)
        # print(self.y)

        if normalize:
            self.X = self.X.reshape(self.X.shape[1], self.X.shape[0])
            min_X = np.min(self.X,0)  # returns an array of means for each beat
            max_X = np.max(self.X,0)
            self.X = (self.X - min_X)/(max_X-min_X)
            min_y = np.min(self.y)
            max_y = np.max(self.y)
            self.y = (self.y - min_y)/(max_y-min_y)
            
            # #UNCOMMENT IN CASE OF OVERFITTING 1 SAMPLE
            # min_X = np.min(self.X)  
            # max_X = np.max(self.X)
            # self.X = (self.X - min_X)/(max_X-min_X)
            # print(self.X)
            # self.X = self.X.reshape(1, 1, 1200)

            
        # reshape input data
        self.X = self.X.reshape(self.X.shape[1], 1, self.X.shape[0])
        self.y = self.y.reshape(self.y.shape[0],)
        # label encode target and ensure the values are floats
        # self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype(np.float)
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_valid=0.2,n_test=0.2):
        # determine sizes
        test_size = round(n_test * len(self.X))
        valid_size = round(n_valid * len(self.X))
        train_size = len(self.X) - test_size - valid_size
        
        train_set, val_set, test_set = random_split(self, [train_size, valid_size, test_size])
        # calculate the split
        return train_set, val_set, test_set
    
    
    


class HDF5Dataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        # y = torch.from_numpy(y)
        # y = Tensor(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)
                    
                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]
 
 
# prepare the dataset
def prepare_data(path,batch_size,n_valid,n_test):
    # load the dataset
    # dataset = CSVDataset(path, normalize = True)
    dataset = HDF5Dataset(path, recursive=False, load_data=False, data_cache_size=4, transform=None)
    # calculate split
    # train, validation, test = dataset.get_splits()

    # test_size = round(n_test * len(self.X))
    # valid_size = round(n_valid * len(self.X))
    # train_size = len(self.X) - test_size - valid_size
        
    # train_set, val_set, test_set = random_split(self, [train_size, valid_size, test_size])
    
    # # prepare data loaders
    # train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    # valid_dl = DataLoader(validation, batch_size=batch_size, shuffle=True)
    # test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
    loader_params = {'batch_size': batch_size, 'shuffle': True}
    train_dl = DataLoader(dataset, **loader_params)
    
    return train_dl

 
# # train the model with validation
# def train_model(train_dl, valid_dl, model, learning_rate, optimizer, model_name):
#     global start
#     # define the optimization
#     # criterion = BCEWithLogitsLoss()
#     # criterion = nn.MSELoss()
#     criterion = nn.SmoothL1Loss()
#     if optimizer == "adam":
#         optimizer = Adam(model.parameters(), lr=learning_rate)
#     elif optimizer == "sgd":
#         optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#     elif optimizer == "adagrad":
#         optimizer = Adagrad(model.parameters(), lr=learning_rate)
    
#     model = model.float()
#     loss_history = list()
#     valid_loss_history = list()
#     # enumerate epochs
#     start = time.time()
#     for epoch in range(5):
#         s = "Epoch " + str(epoch+1) + "/5"
#         print(s, end="\t")
        
#         train_loss = 0.0
#         # enumerate mini batches
#         for i, (inputs, targets) in enumerate(iter(train_dl)):
#             targets = torch.reshape(targets, (targets.shape[0], 1))
#             # Transfer Data to GPU if available
#             # if torch.cuda.is_available():
#             #     print("GPU found")
#             #     inputs, targets = inputs.cuda(), targets.cuda()
#             # clear the gradients
#             optimizer.zero_grad()
#             # compute the model output
#             yhat = model(inputs.float())
#             # print("Prediction vs Target", round(yhat.item(),1), targets.item(),1, end = "\t")
#             # calculate loss
#             loss = criterion(yhat, targets.float())
#             # credit assignment
#             loss.backward()
#             # update model weights
#             optimizer.step()
#             train_loss += loss
            
#         loss_history.append(train_loss.item())
#         print("train loss: "+str(train_loss.item()),end="\t")
        
#         #VALIDATE NETWORK
#         valid_loss = 0.0
#         #model.eval()     # Optional when not using Model Specific layer
#         for i, (inputs, targets) in enumerate(iter(valid_dl)):
#             targets = torch.reshape(targets, (targets.shape[0], 1))
#             # # Transfer Data to GPU if available
#             # if torch.cuda.is_available():
#             #   inputs, targets = inputs.cuda(), targets.cuda()
#             # Forward Pass
#             yhat = model(inputs.float())
#             # Find the Loss
#             loss = criterion(yhat,targets.float())
#             # Calculate Loss
#             valid_loss += loss
            
#         valid_loss_history.append(valid_loss.item())
#         print("validation loss: "+str(valid_loss.item()),end="\n")
            
    # plt.figure()
    # plt.title("Loss history - "+model_name)
    # plt.xlabel("Epochs")
    # plt.grid()
    # plt.ylabel("SmoothL1Loss")
    # plt.plot(loss_history)
    # plt.savefig("./ResNet_results/loss_" +model_name+".png")
    # plt.show()
    
    # plt.figure()
    # plt.title("Loss history zoom - "+model_name)
    # plt.xlim([25, 300])
    # plt.xlabel("Epochs")
    # plt.grid()
    # plt.ylabel("SmoothL1Loss")
    # plt.plot(loss_history[25:])
    # plt.savefig("./ResNet_results/loss_zoom" +model_name+".png")
    # plt.show()
    
    # plt.figure()
    # plt.title("Validation loss history - "+model_name)
    # plt.xlabel("Epochs")
    # plt.grid()
    # plt.ylabel("SmoothL1Loss")
    # plt.plot(valid_loss_history)
    # plt.savefig("./ResNet_results/val_loss_" +model_name+".png")
    # plt.show()
    
    
    
# train the model without validation

def train_model(train_dl, model, learning_rate, optimizer, model_name):
    global start
    # define the optimization
    # criterion = BCEWithLogitsLoss()
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adagrad":
        optimizer = Adagrad(model.parameters(), lr=learning_rate)
    
    model = model.float()
    loss_history = list()
    valid_loss_history = list()
    # enumerate epochs
    start = time.time()
    for epoch in range(5):
        s = "Epoch " + str(epoch+1) + "/5"
        print(s, end="\t")
        
        train_loss = 0.0
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(iter(train_dl)):
            targets = torch.reshape(targets, (targets.shape[0], 1))
            # only if batchsize = 1 reshape inputs
            inputs = torch.reshape(inputs, (1,1,inputs.shape[1]))
            # Transfer Data to GPU if available
            # if torch.cuda.is_available():
            #     print("GPU found")
            #     inputs, targets = inputs.cuda(), targets.cuda()
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs.float())
            # print("Prediction vs Target", round(yhat.item(),1), targets.item(),1, end = "\t")
            # calculate loss
            loss = criterion(yhat, targets.float())
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            train_loss += loss
            
        loss_history.append(train_loss.item())
        print("train loss: "+str(train_loss.item()),end="\t")
            



# evaluate the model
def evaluate_model(test_dl, model, t, title):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(iter(test_dl)):
        # evaluate the model on the test set
        yhat = model(inputs.float())
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        #yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    y_hat = predictions * (max_y-min_y) + min_y
    y = actuals * (max_y-min_y) + min_y
    # calculate accuracy
    acc = math.sqrt(mean_squared_error(actuals, predictions))
    acc_denorm = math.sqrt(mean_squared_error(y, y_hat))
    r2 = r2_score(y, y_hat)
    
    end = time.time()
    runtime = end - start
    
    # plt.figure()
    # plt.plot(np.linspace(25, 175), np.linspace(25, 175), 'magenta')
    # plt.scatter(y,y_hat, s=5, color='indigo')
    # plt.xlabel("Real")
    # plt.ylabel("Predicted")
    # plt.title("Real vs Predicted " + t + " - " + title)
    # plt.grid()
    # plt.savefig("./ResNet_results/scatter_" +title+t+".png")
    # plt.show()
    
    return acc, acc_denorm, r2, runtime, y-y_hat

 
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat