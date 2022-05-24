import time

from utils import config
from utils import create_dataloader
from torchvision.models import resnet50
from torchvision import transforms
from torch import nn
import os
import torch
from imutils import paths
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# data augmentation
trainTransform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

valTransform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# dataset
trainDS, trainLoader = create_dataloader.get_dataloader(rootDir=config.TRAIN,
                                                        transforms=trainTransform,
                                                        batchSize=config.FINETUNE_BATCH_SIZE)
valDS, valLoader = create_dataloader.get_dataloader(rootDir=config.VAL,
                                                    transforms=valTransform,
                                                    batchSize=config.FINETUNE_BATCH_SIZE,
                                                    shuffle=False)
# load up the ResNet50 model
model = resnet50(pretrained=True)
numFeatures = model.fc.in_features

for module, param in zip(model.modules(), model.parameters()):
    if isinstance(module, nn.BatchNorm2d):
        param.requires_grad = False

# define the network head and attach it to the model
headModel = nn.Sequential(
    nn.Linear(numFeatures, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(trainDS.classes))
)
model.fc = headModel
model = model.to(config.DEVICE)

# initializer loss function and optimizer
lossFunc = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=config.LR)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDS) // config.FINETUNE_BATCH_SIZE
valSteps = len(valDS) // config.FINETUNE_BATCH_SIZE

# initializer dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.EPOCHS)):
    model.train()
    totalTrainLoss = 0
    totalValLoss = 0
    trainCorrect = 0
    valCorrect = 0

    # loop over the training set
    for (i, (x, y)) in enumerate(trainLoader):
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        pred = model(x)
        loss = lossFunc(pred, y)
        loss.backward()
        if (i + 2) % 2 == 0:
            opt.step()
            opt.zero_grad()
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # switch off autograd
    with torch.no_grad():
        model.eval()
        for (x, y) in valLoader:
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = model(x)
            totalValLoss += lossFunc(pred, y)
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    trainCorrect = trainCorrect / len(trainDS)
    valCorrect = valCorrect / len(valDS)

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
        avgValLoss, valCorrect))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time takenn to train the model: {:.2f}s".format(endTime - startTime))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.FINETUNE_PLOT)

# save model
torch.save(model, config.FINETUNE_MODEL)