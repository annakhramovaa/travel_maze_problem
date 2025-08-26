import copy
import random
import numpy.linalg.linalg
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import itertools
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Параметры задачи
L = 2 # Количество шагов
n = 5 # Размер матриц перестановок
batch_size = 32
learn_rate = 0.001
num_epochs = 6
# Генерация датасета

def get_key(dictionary, target_value):
    return next((key for key, value in dictionary.items() if torch.equal(value, target_value)), None)

def create_perms(n):
    perms = list(itertools.permutations(range(n)))
    return perms

def generate_permutation_dataset(perms, n, L):
    perms = [torch.tensor(p) for p in perms]
    dataset = []
    for _ in range(20000):
        inputs = torch.stack([perms[np.random.randint(len(perms))] for _ in range(L)])
        targets = inputs[0]
        for i in range(1, L):
            targets = inputs[i][targets]
        dataset.append((inputs, targets))
    return dataset

class MatrixMultNetwork(nn.Module):
    def __init__(self, n):
        self.n = n
        self.L = L
        super(MatrixMultNetwork, self).__init__()
        self.fc1 = nn.Linear(2 * self.n, 3*self.n**2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3*self.n**2, self.n)

    def forward(self, x):
        x = x.view(-1, 2*self.n)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class DeepMatrixNetwork(nn.Module):
    def __init__(self, n, L):
        super(DeepMatrixNetwork, self).__init__()
        self.n = n
        self.L = L
        self.fc = nn.ModuleList()
        for i in range(self.L-1):
            self.fc.append(nn.Linear(2*self.n, 2*n**4))
            self.fc.append(nn.Linear(2*self.n**4, self.n))

    def forward(self, x):
        X, Y = [], []
        X.append(x[:, :2, :].view(-1, 2*self.n))
        Y.append(torch.relu(self.fc[0](X[0])))
        Y[0] = self.fc[1](Y[0])
        if self.L == 2:
            return Y[0]
        for i in range(1, self.L-1):
            X.append(torch.cat((x[:, i + 1, :], Y[- 1]), dim=1).view(-1, 2 * self.n))
            Y.append(torch.relu(self.fc[2 * i](X[-1])))
            Y[i] = self.fc[2 * i + 1](Y[-1])
        return Y[-1]

class MatrixDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        matrices, target = self.dataset[idx]
        matrices = torch.stack([torch.tensor(matrix, dtype=torch.float32) for matrix in matrices])
        target = torch.tensor(target, dtype=torch.float32)
        return matrices, target

train_loss = []   # Для хранения loss на обучающей выборке
test_loss = []   # Для хранения loss на тестовой выборке
train_acc = []
test_acc = []

# Создание датасета
dataset = generate_permutation_dataset(create_perms(n), n, L)
matrix_dataset = MatrixDataset(dataset)

# Создание DataLoader
dataloader = DataLoader(matrix_dataset, batch_size=batch_size, shuffle=False)

# Создание модели
# shallow_model = MatrixMultNetwork(n).to(device)
shallow_model = DeepMatrixNetwork(n, L).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(shallow_model.parameters(), lr=learn_rate, weight_decay=1e-5)

# делим датасет на тренировочный и тестовый
train_size = int(0.80 * len(matrix_dataset))
test_size = len(matrix_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(matrix_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# training
for epoch in range(num_epochs):

    shallow_model.train()
    cur_train_loss = 0
    cur_test_loss = 0
    acc_train = 0
    acc_test = 0
    for train_inp, train_outp in train_loader:
        train_inp, train_outp = train_inp.to(device), train_outp.to(device)
        optimizer.zero_grad()
        outputs = shallow_model(train_inp)
        loss = criterion(outputs, train_outp)
        loss.backward()
        optimizer.step()
        cur_train_loss += loss.item()
        predicted_vectors, pred_ind = torch.sort(outputs, descending=True)
        true_vectors, true_ind = torch.sort(train_outp, descending=True)
        accuracy = (pred_ind == true_ind).float().mean()
        acc_train += accuracy.item()
    avg_tr_loss = cur_train_loss / len(train_loader)
    train_loss.append(avg_tr_loss)
    accuracy_train = acc_train/len(train_loader)
    train_acc.append(accuracy_train)
    end.record()
    torch.cuda.synchronize()  # Ждем завершения всех операций

    shallow_model.eval()
    with torch.no_grad():
        for test_inp, test_outp in test_loader:
            test_inp, test_outp = test_inp.to(device), test_outp.to(device)
            outputs = shallow_model(test_inp)
            loss = criterion(outputs, test_outp)
            cur_test_loss += loss.item()
            predicted_vectors, pred_ind = torch.sort(outputs, descending=True)
            true_vectors, true_ind = torch.sort(test_outp, descending=True)
            accuracy = (pred_ind == true_ind).float().mean()
            acc_test += accuracy.item()
    avg_test_loss = cur_test_loss / len(test_loader)
    test_loss.append(avg_test_loss)
    accuracy_test = acc_test/len(test_loader)
    test_acc.append(accuracy_test)

    # if epoch % 10 == 0:
    print(f"Epoch {epoch}, Train Loss: {avg_tr_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
              f"Train Accuracy: {accuracy_train:.4f}, Test Accuracy: {accuracy_test:.4f}")

with torch.no_grad():
    for test_inp, test_outp in test_loader:
        test_inp, test_outp = test_inp.to(device), test_outp.to(device)
        outputs = shallow_model(test_inp)
        print("predicted matrix", outputs)
        print("real matrix", test_outp)

print(f"Время выполнения: {start.elapsed_time(end) / 1000:.2f} секунд")
#  Визуализация графиков
plt.figure(1, figsize=(10, 5))
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.grid(True)

plt.figure(2, figsize=(10, 5))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Curve')
plt.legend()
plt.grid(True)
plt.show()