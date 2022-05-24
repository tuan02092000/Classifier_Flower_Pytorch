from utils import config
from utils import create_dataloader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import argparse
import torch

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
args = vars(ap.parse_args())

# build our data pre-processing pipeline
testTransform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
invStd = [1/s for s in config.STD]

# define our de-normalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)

# initialize our test dataset and dataloader
print("[INFO] loading the dataset...")
(testDS, testLoader) = create_dataloader.get_dataloader(config.VAL,
                                                        transforms=testTransform,
                                                        batchSize=config.PRED_BATCH_SIZE,
                                                        shuffle=True)
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = "cpu"

# load the model
print("[INFO] loading the model...")
model = torch.load(args["model"], map_location=map_location)
model.to(config.DEVICE)
model.eval()

# grab a batch of test data
batch = next(iter(testLoader))
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure("Results", figsize=(10, 10))

# switch off autograd
with torch.no_grad():
    images = images.to(config.DEVICE)

    # make the predictions
    print("[INFO] performing inference")
    preds = model(images)

    for i in range(0, config.PRED_BATCH_SIZE):
        ax = plt.subplot(config.PRED_BATCH_SIZE, 1, i + 1)
        image = images[i]
        image = deNormalize(image).cpu().numpy()
        image = (image * 255).astype("uint8")
        image = image.transpose((1, 2, 0))

        # grab the ground truth label
        idx = labels[i].cpu().numpy()
        gtLabel = testDS.classes[idx]

        # grab the predicted label
        pred = preds[i].argmax().cpu().numpy()
        predLabel = testDS.classes[pred]

        # add the result and image to the plot
        infor = "Ground truth: {}, Predicted: {}".format(gtLabel, predLabel)
        plt.imshow(image)
        plt.title(infor)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

