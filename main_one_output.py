import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from ResNet_class import ResNet

############# DATA PREPARATION ###################

dataset = pd.read_csv('./Dataset_QRS_detection/dataset_950.csv',na_filter=False,low_memory=False)

dataset_transpose = dataset.transpose()

#shuffle dataset rows
np.random.seed(11) # random seed grant us reprudicible results
dataset_shuffle = dataset_transpose.sample(frac=1).reset_index(drop=True)

# transform pandas dataframe into numpy array of floats and re-transpose
dataset_array = dataset_shuffle.to_numpy().astype(np.float).transpose()
print("Dataset shape: ",dataset_array.shape)

window = 950

X = dataset_array[:window,:]
y = dataset_array[window,:]
print("Shape X: ",X.shape)
print("Shape y: ",y.shape)


# normalization of the dataset
# Not sure if needed

min_X = np.min(X,0)  # returns an array of means for each beat
max_X = np.max(X,0)
min_y = np.min(y,0)  # returns an array of means for each beat
max_y = np.max(y,0)

X_norm = (X - min_X)/(max_X-min_X)
y_norm = (y - min_y)/(max_y-min_y)
# retranspose
X_norm = X_norm.transpose()
print("New shape X: ", X_norm.shape)

# dataset split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.33, shuffle = False)

# reshape dataset for network input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
y_train = y_train.reshape(y_train.shape[0],1)

################# RESNET ####################
output_directory = "./ResNet_results/"
input_shape = (window,1)
verbose = True
n_classes = 1

model = ResNet(output_directory, input_shape, n_classes, verbose)
model.fit(X_train, y_train)
