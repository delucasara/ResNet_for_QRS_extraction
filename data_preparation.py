import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

def split_dataset(dataset, n_val=0.2, n_test=0.2, suffle = True):
    dataset_len = len(dataset.columns)
    dataset_cols = dataset.columns
    
    train_split = round(dataset_len * (1 - n_val - n_test))
    valid_split = round(dataset_len * n_val) + train_split
    
    train = dataset[dataset_cols[:train_split]]
    valid = dataset[dataset_cols[train_split:valid_split]]
    test = dataset[dataset_cols[valid_split:]]

    return [train, valid, test]
    
df = pd.read_csv("./Dataset_QRS_detection/dataset_different.csv")

columns_prova = df.columns[:300]
df = df[columns_prova]

# maybe you wanna do augmentation here?

df = df.transpose()
df = df.sample(frac=1)
df = df.transpose()

# # normalization
for col in df.columns:
    data = df[col][:1200].to_numpy()
    data = data[np.logical_not(np.isnan(data))]
    min_X = np.min(data)
    max_X = np.max(data)
    data = (data - min_X)/(max_X-min_X)
    df[col][:len(data)] = data

plt.figure()
plt.plot(data)
plt.show()

targets = df.iloc[1200,:].to_numpy()
min_y = np.min(targets)
max_y = np.max(targets)
targets = (targets - min_y)/(max_y-min_y)
df.iloc[1200,:] = targets



sets = split_dataset(df)
sets_names = ["train", "validation", "test"]


for i, item in enumerate(sets):
    file_name = './Prove/' + sets_names[i] + '/' + sets_names[i] + '.h5'
    hf = h5py.File(file_name, 'w')
    for col in item.columns:
        data = df[col][:1200].to_numpy()
        data = data[np.logical_not(np.isnan(data))]
        label = df[col][1200]
        group = hf.create_group(col)
        group.create_dataset('data',data=data)
        group.create_dataset('label',data=label)
        
    hf.close()
    


    
    