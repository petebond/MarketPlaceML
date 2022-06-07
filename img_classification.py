# %%
from __future__ import print_function, division
from pyexpat import model
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True
plt.ion()   # interactive mode
writer = SummaryWriter()


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'img_classes_split'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

def createConfusionMatrix(loader, model, class_names):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # constant for classes
    classes = class_names

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 13, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()


#imshow(out, title=[class_names[x] for x in classes])
#input("Displaying a sample batch - Press Enter to continue...")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, batch_size=4):
    """_summary_

    Args:
        model (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        scheduler (_type_): _description_
        num_epochs (int, optional): _description_. Defaults to 25.
        batch_size (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print("Training...")
                model.train()  # Set model to training mode
            elif phase == 'val':
                print("Validating...")
                model.eval()   # Set model to evaluate mode
            #else:
            #   print("Testing...")
            #    model.test()

            running_loss = 0.0
            running_corrects = 0
            counter = 0
            hund_loss = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                
                #with torch.set_grad_enabled(phase == 'train'):
                counter += 1
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                hund_loss += loss.item()
                if counter % 100 == 0:
                    print(f'{phase} loss: {(hund_loss / 100):.4f} Running corrects: {running_corrects} Batch No: {counter}')
                # TRAINING LOSS STAT UNCOMMENT BELOW
                    if phase == 'train':
                        writer.add_scalar('Training Loss', (hund_loss / 100), time.time())                      
                    hund_loss = 0

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / (counter * batch_size)

            # EPOCH LOSS STAT UNCOMMENT BELOW
            # writer.add_scalar('Epoch Loss', epoch_loss, epoch)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {(epoch_acc * 100):.1f}%')
            writer.add_scalar('Validation Accuracy', epoch_acc, epoch)
            writer.add_scalar('Validation Loss', epoch_loss, epoch)
            writer.add_figure(
                "Confusion matrix",
                createConfusionMatrix(dataloaders[phase], model, class_names),
                epoch
                )

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save the model to file
                

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {(best_acc * 100):.1f}%')
    
    
    # load best model weights
    print("Saving model...")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'models/model_state_dict.pt')
    return model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = models.resnet50(pretrained=True)
        for i, param in enumerate(self.features.parameters()):
            if i < 47:
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.features.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, len(class_names))
            )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return x
        
# %%
#input("Entering training loop - Press Enter to continue...")

print("""
Choose wisely:
[l]oad model and validate
[t]rain model and validate
""")
choice = input("(l/t): ")
if choice.lower() == 't':

# finetuning the convnet
    batch_size = 8
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    trained_model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=30, batch_size=batch_size)

    
    writer.flush()



# %%
# convnet as fixed feature extractor
elif choice.lower() == 'l':
    batch_size = 8
    model = CNN().to(device)
    model.load_state_dict(torch.load('models/model_state_dict.pt'))
    writer.add_figure(
                "Confusion matrix",
                createConfusionMatrix(dataloaders['val'], model, class_names),
                0
                )
    writer.flush()


# %%
