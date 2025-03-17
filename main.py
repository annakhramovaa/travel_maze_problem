import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F


class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for img_name in os.listdir(data_dir):
            self.image_paths.append(os.path.join(data_dir, img_name))
            l_bracket = self.image_paths[-1].find('[')
            r_bracket = self.image_paths[-1].find(']')
            label_str = self.image_paths[-1][l_bracket + 1: r_bracket]
            label_str = label_str.replace('.', '')
            targets = list(map(int, label_str.split()))
            targets = [x - 1 for x in targets]
            targets = torch.tensor(targets, dtype=torch.int64)
            targets = F.one_hot(targets, 3)
            targets = targets.float()
            self.labels.append(targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("1")
        if self.transform:
            image = self.transform(image)

        return image, label

# 4. Определение однослойной нейронной сети
class PermutationNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PermutationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 150)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        y1 = x[:, 0:3]
        y2 = x[:, 3:6]
        y3 = x[:, 6:9]
        y1 = F.softmax(y1, dim=1)
        y2 = F.softmax(y2, dim=1)
        y3 = F.softmax(y3, dim=1)
        y = torch.cat((y1, y2, y3), dim=1)
        y = y.reshape(len(y), 3, 3)
        return y
# 5. Параметры задачи
n = 9
image_size = 16  # Размер изображения
learning_rate = 1e-6
epochs = 10
batch_size = 100
test_batch_size = 100 #  Размер тестовой выборки

# 6. Создание экземпляра модели
model = PermutationNet(image_size**2, n)

# 7. Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


dataset = TestDataset(data_dir='D:\\учеба\\maze_problem\\Dataset', transform=transforms.Compose(
    [transforms.Resize((image_size, image_size)),
    transforms.ToTensor()]))
train_size = int(0.80 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 10. Обучение модели
train_losses = []  # Для хранения loss на обучающей выборке
test_losses = []   # Для хранения loss на тестовой выборке

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()  # Обнуляем градиенты
        outputs = model(data)  # Вычисляем выход
        loss = criterion(outputs, targets)  # Вычисляем loss
        loss.backward()  # Обратное распространение ошибки
        optimizer.step()  # Обновляем веса
        running_loss += loss.item()  # Записываем loss для обучающей выборки

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    # Оценка на тестовой выборке
    running_test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
             test_outputs = model(x) # Вычисляем выход на тестовой выборке
             test_loss = criterion(test_outputs, y)  # Вычисляем loss на тестовой выборке
             running_test_loss += test_loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(test_loss.item()) # Записываем loss на тестовой выборке

# 11. Тестирование модели (вывод результатов)
model.eval()

with torch.no_grad():
    for x, y in test_loader:
        test_output = model(x)
        print("Original:\n", y)
        print("Predicted:\n", test_output)
        print('-----')

# 12. Визуализация графиков
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
