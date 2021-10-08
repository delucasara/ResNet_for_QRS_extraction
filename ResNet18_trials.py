from QRS_detection import *
from ResNet1d import *
from torchsummary import summary
import matplotlib.pyplot as plt


model = ResNet18_Counter()
path = './Dataset_QRS_detection/dataset_prepared_augmented_reduced.csv'
model_name = "ResNet18"


batch_sizes = [8,16,32]
learning_rates = [1e-2,1e-3,1e-4]
optimizer = ["adam","sgd"]

# optimizer, learning rate, batch size
results = np.zeros((2,3,3),dtype=float)

for i, b in enumerate(batch_sizes):
    print("Batch size: ",b)
    for j, l in enumerate(learning_rates):
        print("Learning rate: ",l)
        for k, o in enumerate(optimizer):
            print("Optimizer: ",o)
            train_dl, valid_dl, test_dl = prepare_data(path,b)
            train_model(train_dl, valid_dl , model, l, o, model_name)
            acc_train, acc_denorm_train, r2_train, runtime, error_train = evaluate_model(train_dl, model, "train", model_name)
            acc_val, acc_denorm_val, r2_val, runtime, error_val = evaluate_model(valid_dl, model, "validation", model_name)
            acc_test, acc_denorm_test, r2_test, runtime_test, error_test = evaluate_model(test_dl, model, "test", model_name)
            results[k][j][i] = acc_denorm_train
 
min_combo = np.argwhere(results == np.min(results))
print("Best combination is: ", optimizer[min_combo[0][0]], learning_rates[min_combo[0][1]], batch_sizes[min_combo[0][2]])











