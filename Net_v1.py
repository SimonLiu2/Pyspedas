import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(5, 60)
        self.hidden2 = nn.Linear(60, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.out(x)
        return x
    

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_model(model, train_dataloader, val_dataloader, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.apply(init_weights)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader)}, Val Loss: {val_loss/len(val_dataloader)}')

def test_model(model, test_dataloader):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    print(f'Test Loss: {test_loss/len(test_dataloader)}')

def predict(model, dataloader):
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, _ = data
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
    return predictions


def load_data(batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('dataset_preprocessed.csv')
    cols = df.columns
    array_dataset_preprocessed = df[cols[:207]].to_numpy()
    array_dataset_preprocessed[:,-1] = np.log10(array_dataset_preprocessed[:,-1])
    array_dataset_preprocessed = df.to_numpy()


    # Separate features and labels
    features = array_dataset_preprocessed[:, [143,173,203,204,205]]
    labels = array_dataset_preprocessed[:, -1]

    # Create TensorDataset
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32, device=device, requires_grad=False), 
                            torch.tensor(labels, dtype=torch.float32, device=device,requires_grad=False))

    # Using 5-fold cross-validation
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Continue to split train_dataset into 5 folds
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    return train_dataloader, val_dataloader, test_dataloader  

