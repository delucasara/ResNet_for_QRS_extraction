from QRS_detection import *
from ResNet1d import *
from torchsummary import summary
import matplotlib.pyplot as plt



def run_model(path, model, model_name):
    train_dl, valid_dl, test_dl = prepare_data(path,batch_size=16)
    # train the model
    train_model(train_dl, valid_dl, model, 1e-3, "adam", model_name)
    # evaluate the model on training
    acc_train, acc_denorm_train, r2_train, runtime, error_train = evaluate_model(train_dl, model, "train", model_name)
    print("")
    print('Train RMSE: %.3f' % acc_train)
    print('Train RMSE (denorm): %.3f' % acc_denorm_train)
    print('Train R2 score: %.3f' %r2_train)
    # acc_test=0
    # evaluate model on test
    acc_test, acc_denorm_test, r2_test, runtime_test, error_test = evaluate_model(test_dl, model, "test", model_name)
    print('Test RMSE: %.3f' % acc_test)
    print('Test RMSE (denorm): %.3f' % acc_denorm_test)
    print('Test R2 score: %.3f' %r2_test)
    print("Execution time: %.3f" %runtime)
    
    plt.figure()
    plt.hist(error_train, 50, color='dodgerblue')
    plt.hist(error_test, 50, color='gold')
    plt.xlabel('Error on estimation')
    plt.ylabel('Frequency')
    plt.legend(["Train","Test"])
    plt.title('Histogram for the estimation error - ' + model_name)
    plt.savefig("./ResNet_results/hist_" +model_name+".png")
    plt.show()
    
    return acc_train, acc_denorm_train, r2_train, acc_test, acc_denorm_test, r2_test, runtime
    

path = './Dataset_QRS_detection/dataset_prepared_augmented.csv'
model_names = ['ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'All'] 

for i in range(len(model_names)):
    print(str(i+1)+ ". "+str(model_names[i]))
model_name = int(input("Choose the model to train (1-7): "))

if model_name != 7:
    if model_name == 1:
        model = ResNet10_Counter()
        print(summary(model, (1,1200)))
    elif model_name == 2:
        model = ResNet18_Counter()
        print(summary(model, (1,1200)))
    elif model_name == 3:
        model = ResNet34_Counter()
        print(summary(model, (1,1200)))
    elif model_name == 4:
        model = ResNet50_Counter()
        print(summary(model, (1,1200)))
    elif model_name == 5:
        model = ResNet101_Counter()
        print(summary(model, (1,1200)))
    elif model_name == 6:
        model = ResNet152_Counter()
        print(summary(model, (1,1200)))

    acc_train, acc_denorm_train, r2_train, acc_test, acc_denorm_test, r2_test, runtime = run_model(path, model, model_names[model_name-1])

    
else:
    accuracy = {}
    for i in range(1,7):
        if i == 1:
            model = ResNet10_Counter()
            print("\n--------------------")
            print("Training",model_names[i-1])
            acc_train, acc_denorm_train, r2_train, acc_test, acc_denorm_test, r2_test, runtime = run_model(path, model, model_names[i-1])
            accuracy[model_names[i-1]] = {"train": {"MSE": acc_train,
                                                    "MSE denorm": acc_denorm_train,
                                                    "R2": r2_train},
                                          "test": {"MSE": acc_test,
                                                    "MSE denorm": acc_denorm_test,
                                                    "R2": r2_test},
                                          "time": runtime}
            
        elif i == 2:
            model = ResNet18_Counter()
            print("\n--------------------")
            print("Training",model_names[i-1])
            acc_train, acc_denorm_train, r2_train, acc_test, acc_denorm_test, r2_test, runtime = run_model(path, model, model_names[i-1])
            accuracy[model_names[i-1]] = {"train": {"MSE": acc_train,
                                                    "MSE denorm": acc_denorm_train,
                                                    "R2": r2_train},
                                          "test": {"MSE": acc_test,
                                                    "MSE denorm": acc_denorm_test,
                                                    "R2": r2_test},
                                          "time": runtime}
        elif i == 3:
            model = ResNet34_Counter()
            print("\n--------------------")
            print("Training",model_names[i-1])
            acc_train, acc_denorm_train, r2_train, acc_test, acc_denorm_test, r2_test, runtime = run_model(path, model, model_names[i-1])
            accuracy[model_names[i-1]] = {"train": {"MSE": acc_train,
                                                    "MSE denorm": acc_denorm_train,
                                                    "R2": r2_train},
                                          "test": {"MSE": acc_test,
                                                    "MSE denorm": acc_denorm_test,
                                                    "R2": r2_test},
                                          "time": runtime}
        elif i == 4:
            model = ResNet50_Counter()
            print("\n--------------------")
            print("Training",model_names[i-1])
            acc_train, acc_denorm_train, r2_train, acc_test, acc_denorm_test, r2_test, runtime = run_model(path, model, model_names[i-1])
            accuracy[model_names[i-1]] = {"train": {"MSE": acc_train,
                                                    "MSE denorm": acc_denorm_train,
                                                    "R2": r2_train},
                                          "test": {"MSE": acc_test,
                                                    "MSE denorm": acc_denorm_test,
                                                    "R2": r2_test},
                                          "time": runtime}
        elif i == 5:
            model = ResNet101_Counter()
            print("\n--------------------")
            print("Training",model_names[i-1])
            acc_train, acc_denorm_train, r2_train, acc_test, acc_denorm_test, r2_test, runtime = run_model(path, model, model_names[i-1])
            accuracy[model_names[i-1]] = {"train": {"MSE": acc_train,
                                                    "MSE denorm": acc_denorm_train,
                                                    "R2": r2_train},
                                          "test": {"MSE": acc_test,
                                                    "MSE denorm": acc_denorm_test,
                                                    "R2": r2_test},
                                          "time": runtime}
        elif i == 6:
            model = ResNet152_Counter()
            print("\n--------------------")
            print("Training",model_names[i-1])
            acc_train, acc_denorm_train, r2_train, acc_test, acc_denorm_test, r2_test, runtime = run_model(path, model, model_names[i-1])
            accuracy[model_names[i-1]] = {"train": {"MSE": acc_train,
                                                    "MSE denorm": acc_denorm_train,
                                                    "R2": r2_train},
                                          "test": {"MSE": acc_test,
                                                    "MSE denorm": acc_denorm_test,
                                                    "R2": r2_test},
                                          "time": runtime}
    np.save('./ResNet_results/accuracy.npy', accuracy, allow_pickle=True)
    print(accuracy)
