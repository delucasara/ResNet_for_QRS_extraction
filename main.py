from QRS_detection import *
from ResNet1d import *
from torchsummary import summary
import matplotlib.pyplot as plt

path = './Dataset_QRS_detection/dataset_prepared_augmented_reduced.csv'
model_name = "ResNet18"

batch_size=8
learning_rate = 1e-4
optimizer = "adam"

model = ResNet18_Counter()
train_dl, valid_dl, test_dl = prepare_data(path,batch_size)
# train the model
train_model(train_dl, valid_dl, model, learning_rate, optimizer, model_name)
# evaluate the model on training
acc_train, acc_denorm_train, r2_train, runtime, error_train = evaluate_model(train_dl, model, "train", model_name)

print('Train RMSE: %.3f' % acc_denorm_train)
print('Train R2 score: %.3f' %r2_train)

acc_val, acc_denorm_val, r2_val, runtime, error_val = evaluate_model(valid_dl, model, "validation", model_name)
print('Validation RMSE: %.3f' % acc_denorm_val)
print('Validation R2 score: %.3f' %r2_val)

# evaluate model on test
acc_test, acc_denorm_test, r2_test, runtime_test, error_test = evaluate_model(test_dl, model, "test", model_name)
print('Test RMSE: %.3f' % acc_denorm_test)
print('Test R2 score: %.3f' %r2_test)
print("Execution time: %.3f" %runtime)

plt.figure()
plt.hist(error_train, 50, color='dodgerblue')
plt.hist(error_val, 50, color='pink')
plt.hist(error_test, 50, color='gold')
plt.xlabel('Error on estimation')
plt.ylabel('Frequency')
plt.legend(["Train", "Validation","Test"])
plt.title('Histogram for the estimation error - ' + model_name)
plt.savefig("./ResNet_results/hist_" +model_name+".png")
plt.show()