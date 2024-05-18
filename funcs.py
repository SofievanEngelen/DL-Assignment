# Function for model evaluation
import numpy as np
import torch
from bayes_opt import BayesianOptimization
from torch import nn
from torch.utils.data import DataLoader

from model import CNN
from preprocessing import AudioDataset


def evaluate_model(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            total_loss += loss.item()
            count += 1
    average_loss = total_loss / count
    return average_loss


# Defining early stopping class, used in tuning
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Defining training and evaluation function
def train_and_evaluate(lr: float,
                       batch_size: int,
                       tds: AudioDataset,
                       vds: AudioDataset,
                       num_epochs=10,
                       criterion=nn.MSELoss(),
                       accumulation_steps=4):
    # Load the datasets
    train_dataloader = DataLoader(tds, batch_size=int(batch_size), shuffle=True, num_workers=4)
    val_dataloader = DataLoader(vds, batch_size=int(batch_size), shuffle=False, num_workers=4)

    # Initialise the CNN model, its optimisers and the scheduler
    input_size = np.array(tds.features).shape[1]
    model = CNN(input_size=input_size)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=2, factor=0.5, min_lr=0.0001)

    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(num_epochs):
        model.train()
        optimiser.zero_grad()
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.float()
            labels = labels.float()

            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimiser.step()
                optimiser.zero_grad()

        validation_loss = evaluate_model(model, criterion, val_dataloader)
        scheduler.step(validation_loss)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), 'best_model.pth')

        early_stopping(validation_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_val_loss, best_val_loss


# Bayesian optimization for hyperparameter tuning
def tune_hyperparameters(train_dataset, val_dataset):
    def objective(learning_rate, batch_size):
        target, mse = train_and_evaluate(learning_rate, batch_size, train_dataset, val_dataset)
        print(f"Target: {target}, MSE: {mse}")
        return target

    pbounds = {'learning_rate': (0.0001, 0.005), 'batch_size': (16, 128)}

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=5,
    )

    # Extracting the best parameters and their corresponding MSE
    best_params = optimizer.max['params']
    _, best_mse = train_and_evaluate(lr=best_params['learning_rate'],
                                     batch_size=best_params['batch_size'],
                                     tds=train_dataset,
                                     vds=val_dataset)

    return optimizer.max, best_mse
