#%%
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage import io
import torchvision.transforms as transforms
import torch.nn.functional as F

class ProductImageCategoryDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X = pd.read_pickle('models/image_model_X.pkl')
        self.X['image'] = self.X.values.tolist()
        self.X = self.X['image']
        self.y = pd.read_pickle('models/image_model_y.pkl')
        assert len(self.X) == len(self.y)

    def __getitem__(self, index):
        features = self.X.iloc[index]
        label = self.y.iloc[index]
        features = torch.tensor(features).float()
        features = features.reshape(3, 64, 64)
        label = int(label)
        return (features, label)

    def __len__(self):
        return len(self.X)

# %%
dataset = ProductImageCategoryDataset()
# print(len(dataset))
# dataset[10187]

# %%
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
# %%
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(2704, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 14)
        torch.nn.Softmax(dim=1)        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x
# %%

model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
writer = SummaryWriter()
writer2 = SummaryWriter()



# %%
num_epochs = 100
batch_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader, 0):
        # print(batch)
        features, labels = batch
        # zero the gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # write to tensorboard
        writer.add_scalar('Loss', loss, batch_idx)
        running_loss += loss.item()
        batch_losses.append(loss.item())
        if batch_idx % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 200))
            running_loss = 0.0
        #break
    avg_loss = sum(batch_losses[-3:])/3
    print('epoch: %d, avg_loss: %.3f' % (epoch + 1, avg_loss))
    writer2.add_scalar('Avg Loss', avg_loss, epoch)
writer.flush()
writer2.flush()

# %% 
print(outputs)