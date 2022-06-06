# %%
from __future__ import print_function, division
from pyexpat import model

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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
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

imshow(out, title=[class_names[x] for x in classes])
input("Displaying a sample batch - Press Enter to continue...")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, batch_size=4):
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
            # input("EPOCH COMPLETE: Press Enter to continue...")

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

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def test_species(model, len_labels): 
    # Load the model that we saved at the end of the training loop   
     
    labels_length = len_labels
    labels_correct = list(0. for i in range(labels_length)) # list to calculate correct labels
    labels_total = list(0. for i in range(labels_length))   # list to keep the total # of labels per type
  
    with torch.no_grad(): 
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
             
            label_correct_running = (preds == outputs).squeeze() 
            label = outputs[0] 
            if label_correct_running.item():  
                labels_correct[label] += 1 
            labels_total[label] += 1  
  
    label_list = list(labels.keys()) 
    for i in range(labels_length): 
        print("THIS IS NEW! IF YOU SEE THIS, YOU ARE GOOD")
        print('Accuracy to predict %5s : %2d %%' % (label_list[i], 100 * labels_correct[i] / labels_total[i])) 

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

choice = input("Train the pre-trained model or Fixed feature extractor? (t/f): ")
if choice == 't':

# finetuning the convnet
    batch_size = 16
    model = CNN().to(device)
    model_ft = model.features 
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=20, batch_size=batch_size)

    visualize_model(model_ft)
    input("visualising model - Press Enter to continue...")
    test_species(model_ft, batch_size)
    writer.flush()
    plt.ioff()
    plt.show()


# %%
# convnet as fixed feature extractor
elif choice == 'f':
    model_conv = torchvision.models.resnet50(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                        num_epochs=25)

    visualize_model(model_conv)
    input("visualising model - Press Enter to continue...")
    plt.ioff()
    plt.show()

# %%
