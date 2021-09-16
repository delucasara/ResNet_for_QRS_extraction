import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from ResNet_class import ResNet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

############# DATA PREPARATION ###################

dataset = pd.read_csv('./Dataset_QRS_detection/dataset_950.csv',na_filter=False,low_memory=False)

dataset_transpose = dataset.transpose()

#shuffle dataset rows
np.random.seed(11) # random seed grant us reprudicible results
dataset_shuffle = dataset_transpose.sample(frac=1).reset_index(drop=True)

# transform pandas dataframe into numpy array of floats and re-transpose
dataset_array = dataset_shuffle.to_numpy().astype(np.float).transpose()
print("Dataset shape: ",dataset_array.shape)

beat_len = 950

X = dataset_array[:beat_len,:]
y = dataset_array[beat_len,:]
print("Shape X: ",X.shape)
print("Shape y: ",y.shape)

#transform y to categorical for classification problem
n_classes = int(np.amax(y)) + 1
print("Number of classes: ",n_classes)
y_categ = to_categorical(y, n_classes)
print("New shape y:",y_categ.shape)

# normalization of the dataset
# Not sure if needed

min_X = np.min(X,0)  # returns an array of means for each beat
max_X = np.max(X,0)

X_norm = (X - min_X)/(max_X-min_X)
#reshape
X_norm = X_norm.transpose()
print("New shape X: ", X_norm.shape)

# dataset split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_categ, test_size=0.33, shuffle = False)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

################# RESNET ####################
output_directory = "./ResNet_class_results/"
input_shape = (beat_len,1)
verbose = True

model = ResNet(output_directory, input_shape, n_classes, verbose)
model.fit(X_train, y_train)

y_hat = model.predict(X_test)

testRMSE = math.sqrt(mean_squared_error(y_test, y_hat))
testMAPE = mean_absolute_percentage_error(y_test, y_hat)

print("Test Score:")
print('%.2f RMSE' % (testRMSE))
print('%.2f MAPE' % (testMAPE))







