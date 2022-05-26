#%%
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from skimage import io
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


# %%

class ProductImageCategoryDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self):
        """_summary_
        """
        super().__init__()
        self.X = pd.read_pickle('models/image_model_X.pkl')
        self.y = pd.read_pickle('models/image_model_y.pkl')
        assert len(self.X) == len(self.y)

    def __getitem__(self, index):
        features = self.X.iloc[index]
        label = self.y.iloc[index]
        features = torch.tensor(features)
        features = features.reshape(3, 64, 64)
        label = int(label)
        return (features, label)

    def __len__(self):
        return len(self.X)


# %%
dataset = ProductImageCategoryDataset()
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)


# %%
class CNN(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(2704, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 14)
        torch.nn.Softmax(dim=1)        

    def forward(self, x):
        """
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x


class resnet50(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = models.resnet50(pretrained=True)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(7*7*2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 13),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1, 7*7*2048)
        x = self.regressor(x)
        return x

    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

def train_model(model, criterion, optimiser, writer, writer2, num_epochs=4):
    batch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, 0):
            # print(batch)
            features, labels = batch
            # forward + backward + optimize
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            # write to tensorboard
            writer.add_scalar('Loss', loss, batch_idx)
            running_loss += loss.item()
            batch_losses.append(loss.item())
            # zero the gradients
            optimiser.zero_grad()
            
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
batch_size = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = dataset
train_loader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
classes = ("Other Goods",
           "Video Games & Consoles",
           "Health & Beauty",
           "Computers & Software",
           "Office Furniture & Equipment",
           "Appliances",
           "Phones, Mobile Phones & Telecoms",
           "DIY Tools & Materials",
           "Clothes, Footwear & Accessories",
           "Sports, Leisure & Travel",
           "Baby & Kids Stuff",
           "Music, Films, Books & Games",
           "Home & Garden")
# %%
def prep_img(img):
    img =img.reshape(64, 64, 3)
    img = (np.transpose(img, (2, 1, 0)))
    return img
# %%
dataiter = iter(train_loader)
images, labels = dataiter.next()
img_list = [x for x in images]
trans_imgs = []
for idx, img in enumerate(img_list):
    img = prep_img(img)
    img_list[idx] = img

image_grid = make_grid(img_list)
img = transforms.ToPILImage()(image_grid)

# display labels for images
print('\n'.join('%5s' % classes[labels[j]] for j in range(batch_size)))
img.show()

  



 # %%
model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
writer = SummaryWriter()
writer2 = SummaryWriter()

train_model(model, criterion, optimiser, writer, writer2, num_epochs=1) 

# %%

model = resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
writer = SummaryWriter()
writer2 = SummaryWriter()

train_model(model, criterion, optimiser, writer, writer2, num_epochs=4) 

# %%
