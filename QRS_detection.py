from numpy import vstack
from torchsummary import summary
import numpy as np
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import SGD, Adam
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
    
 
 
# prepare the dataset
def prepare_data(path,batch_size):
    # load the dataset
    dataset = CSVDataset(path, normalize = True)
    # calculate split
    train, validation, test = dataset.get_splits()
    
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=False)
    valid_dl = DataLoader(validation, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
 
    return train_dl, valid_dl, test_dl

 
# train the model
def train_model(train_dl, valid_dl, model, learning_rate, optimizer, model_name):
    global start
    # define the optimization
    # criterion = BCEWithLogitsLoss()
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    model = model.float()
    loss_history = list()
    valid_loss_history = list()
    # enumerate epochs
    start = time.time()
    for epoch in range(150):
        s = "Epoch " + str(epoch+1) + "/150"
        print(s, end="\t")
        
        train_loss = 0.0
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(iter(train_dl)):
            targets = torch.reshape(targets, (targets.shape[0], 1))
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                print("GPU found")
                inputs, targets = inputs.cuda(), targets.cuda()
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
            
        loss_history.append(train_loss)
        print("train loss: "+str(train_loss.item()),end="\t")
        
        #VALIDATE NETWORK
        valid_loss = 0.0
        #model.eval()     # Optional when not using Model Specific layer
        for i, (inputs, targets) in enumerate(iter(valid_dl)):
            targets = torch.reshape(targets, (targets.shape[0], 1))
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), targets.cuda()
            # Forward Pass
            yhat = model(inputs.float())
            # Find the Loss
            loss = criterion(yhat,targets.float())
            # Calculate Loss
            valid_loss += loss
            
        valid_loss_history.append(valid_loss)
        print("validation loss: "+str(valid_loss.item()),end="\n")
            
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
    # plt.xlim([25, 200])
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