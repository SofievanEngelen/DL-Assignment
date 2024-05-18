# This is a sample Python script.
from funcs import tune_hyperparameters
from preprocessing import preprocessDatasets

# Define paths
smalltrain_path = '/Users/sofie/Downloads/1000-data/'
train_path = '/Users/sofie/Downloads/train/'
test_path = '/Users/sofie/Downloads/test/'


def main():
    train_dataset, val_dataset = preprocessDatasets(smalltrain_path)
    print(train_dataset)
    result, best_mse = tune_hyperparameters(train_dataset, val_dataset)
    print(result)
    print(f"Best MSE: {best_mse}")


if __name__ == '__main__':
    main()
