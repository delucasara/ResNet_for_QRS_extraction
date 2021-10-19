import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import neurokit2 as nk
import statistics
import random


dataset_original = pd.read_csv("./Dataset_QRS_detection/dataset_ratio.csv")
# dataset_original = dataset_original[["P1b1l1","P1b1l2"]]
augmented_dataset = pd.DataFrame()

snr = []
dataset = dataset_original.copy(deep=True)
# adding noise and artifacts
for col in dataset.columns: 
    noise_factor = np.random.uniform(0.05,0.15)
    column_total = dataset[col].values
    beat = column_total[:-1]
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
    column_total[:-1] = beat_noise
    augmented_dataset[col_name] = column_total
    
    #signal to noise
    noise = np.subtract(abs(beat_noise), abs(dataset_original[col][:-1].values))
    ratio = []
    for i in range(len(noise)):
        d = abs(beat_noise[i]) / noise[i] if noise[i] != 0 else 0
        ratio.append(d)
    snr.append(round(statistics.mean(ratio),1))
   
print("Average SNR: ",round(statistics.mean(snr),1))

# plt.figure()
# plt.title("Electrode Motion Artifacts")
# plt.xlabel("Samples")
# plt.ylabel("mV")
# plt.plot(augmented_dataset["P1b1l1_N"][:1200].values, color = "firebrick")    
# plt.show()

# plt.figure()
# plt.title("Powerline Interference")
# plt.xlabel("Samples")
# plt.ylabel("mV")
# plt.plot(augmented_dataset["P1b1l1_N"][:1200].values, color = "forestgreen")
# plt.show()

# plt.figure()
# plt.title("Electromyographic Noise")
# plt.xlabel("Samples")
# plt.ylabel("mV")
# plt.plot(augmented_dataset["P1b1l1_N"][:1200].values, color = "darkorange")
# plt.show()


dataset = dataset_original.copy(deep=True)
# shifting beats
for col in dataset.columns:
    # shift right
    column_total = dataset[col]
    beat = column_total[:-1]
    
    choice = random.choice([0,1])
    if choice == 1:
        shift = np.random.randint(800,900)
    else:
        shift = np.random.randint(100,200)

    Rshift_beat = pd.concat((beat[shift:], beat[:shift]), axis = 0, ignore_index=True)
    col_name = col + "_S"
    column_total[:-1] = Rshift_beat
    augmented_dataset[col_name] = column_total



# dataset = dataset_original.copy(deep=True)
# # shifting beats
# for col in dataset.columns:
#     # shift right
#     column_total = dataset[col]
#     beat = column_total[:-1]
    
#     shift = np.random.randint(100,200)
#     #shift left
#     Lshift_beat = pd.concat((beat[shift:], beat[:shift]), axis = 0, ignore_index=True)
#     col_name = col + "_LS"
#     column_total[:-1] = Lshift_beat
#     augmented_dataset[col_name] = column_total
    


    
augmented_dataset = pd.concat((dataset_original,augmented_dataset), axis = 1)

augmented_dataset = augmented_dataset.transpose()
augmented_dataset = augmented_dataset.sample(frac=1)
augmented_dataset = augmented_dataset.transpose()

augmented_dataset.to_csv("./Dataset_QRS_detection/dataset_ratio_augmented.csv", index = False, header=True)

    
# plt.figure()
# plt.plot(augmented_dataset["P1b1l1_S"][:1200].values)
# plt.plot(augmented_dataset["P1b1l1"][:1200].values)
# plt.show()

# fig, axs = plt.subplots(2, 2)
# fig.suptitle('Dataset augmentation')
# axs[0, 0].plot(augmented_dataset["P1b1l1"][:1200].values)
# axs[0, 0].set_title('Original signal')
# axs[0, 1].plot(augmented_dataset["P1b1l1_N"][:1200].values, 'tab:orange')
# axs[0, 1].set_title('Signal with powerline noise and artifacts')
# axs[1, 0].plot(augmented_dataset["P1b1l1_RS"][:1200].values, 'tab:green')
# axs[1, 0].set_title('Right shifted signal')
# axs[1, 1].plot(augmented_dataset["P1b1l1_LS"][:1200].values, 'tab:red')
# axs[1, 1].set_title('Left shifted signal')

# for ax in axs.flat:
#     ax.set(xlabel='Samples', ylabel='mV')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
    
    
#signal to noise ratio calculation
# noise = np.subtract(augmented_dataset["P1b1l1_N"][:-1].values, augmented_dataset["P1b1l1"][:-1].values)
# snr = []
# for i in range(len(noise)):
#     d = augmented_dataset["P1b1l1_N"][i] / noise[i] if noise[i] != 0 else 0
#     snr.append(d)
# print("Average SNR: ",round(statistics.mean(snr),1))




