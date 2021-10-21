import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import statistics
import random

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

columns_prova = df.columns[:1500]
df = df[columns_prova]

# dataset augmentation
augmented_dataset = pd.DataFrame()

snr = []
dataset = df.copy(deep=True)
# adding noise and artifacts
for col in dataset.columns: 
    noise_factor = np.random.uniform(0.05,0.15)
    column_total = dataset[col].to_numpy()
    beat = column_total[:-1]
    beat = beat[np.logical_not(np.isnan(beat))]
    beat_noise = nk.signal_distort(beat,
                            sampling_rate=1000,
                            powerline_amplitude=0.5, powerline_frequency=50,
                            artifacts_amplitude=1.1, artifacts_frequency=10,
                            artifacts_number=np.random.randint(5,10)
                            )
    # The amplitude of the artifacts (relative to the standard deviation of the signal).
    gauss_noise = np.random.normal(0, 1, beat.shape)
    beat_noise = beat_noise + noise_factor * gauss_noise
    col_name = col + "_N"
    column_total[:beat.shape[0]] = beat_noise
    augmented_dataset[col_name] = column_total
    
    #signal to noise
    noise = np.subtract(abs(beat_noise), abs(beat))
    ratio = []
    for i in range(len(noise)):
        d = abs(beat_noise[i]) / noise[i] if noise[i] != 0 else 0
        ratio.append(d)
    snr.append(round(statistics.mean(ratio),1))
   
print("Average SNR: ",round(statistics.mean(snr),1))


dataset = df.copy(deep=True)
# shifting beats
for col in dataset.columns:
    # shift right
    column_total = dataset[col].to_numpy()
    beat = column_total[:-1]
    beat = beat[np.logical_not(np.isnan(beat))]
    
    choice = random.choice([0,1])
    if choice == 1:
        shift = np.random.randint(800,900)
    else:
        shift = np.random.randint(100,200)

    Rshift_beat = np.concatenate((beat[shift:], beat[:shift]), axis = None)
    col_name = col + "_S"
    column_total[:beat.shape[0]] = Rshift_beat
    augmented_dataset[col_name] = column_total


augmented_dataset = pd.concat((df,augmented_dataset), axis = 1)
augmented_dataset = augmented_dataset.transpose()
augmented_dataset = augmented_dataset.sample(frac=1)
augmented_dataset = augmented_dataset.transpose()


# # normalization
for col in augmented_dataset.columns:
    data = augmented_dataset[col][:1200].to_numpy()
    data = data[np.logical_not(np.isnan(data))]
    min_X = np.min(data)
    max_X = np.max(data)
    data = (data - min_X)/(max_X-min_X)
    augmented_dataset[col][:len(data)] = data

# plt.figure()
# plt.plot(data)
# plt.show()

targets = augmented_dataset.iloc[1200,:].to_numpy()
min_y = np.min(targets)
max_y = np.max(targets)
targets = (targets - min_y)/(max_y-min_y)
augmented_dataset.iloc[1200,:] = targets



sets = split_dataset(augmented_dataset)
sets_names = ["train", "validation", "test"]


for i, item in enumerate(sets):
    file_name = './Dataset_QRS_detection/' + sets_names[i] + '/' + sets_names[i] + '.h5'
    hf = h5py.File(file_name, 'w')
    for col in item.columns:
        data = augmented_dataset[col][:1200].to_numpy()
        data = data[np.logical_not(np.isnan(data))]
        label = augmented_dataset[col][1200]
        group = hf.create_group(col)
        group.create_dataset('data',data=data)
        group.create_dataset('label',data=label)
        
    hf.close()
    


    
    