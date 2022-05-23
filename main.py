#%%
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
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
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
# %%
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 9),
            torch.nn.Flatten(),  # flatten
            # simplify further with linear layers
            torch.nn.Linear(25088, 13),
            torch.nn.Softmax()  # turn into probabilities
        )

    def forward(self, features):
        return self.layers(features)
# %%

model = CNN()
optimizer = torch.optim.SDG(model.parameters(), lr=0.001)

# %%
for batch_idx, batch in enumerate(train_loader):
    # print(batch)
    features, labels = batch
    prediction = model(features)
    # calculate the loss
    loss = F.cross_entropy(prediction, labels)
    print("Loss: ", loss, "\t Batch: ", batch_idx)
    
    # backpropagate the loss
    loss.backward()
    # take an optimization step
    break
# %% 
print(prediction)