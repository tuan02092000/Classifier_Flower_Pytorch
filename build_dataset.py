from utils import config
from imutils import paths
import numpy as np
import shutil
import os

def copy_images(imagePaths, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for path in imagePaths:
        imageName = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[1]
        labelFolder = os.path.join(folder, label)

        if not os.path.exists(labelFolder):
            os.makedirs(labelFolder)
        destination = os.path.join(labelFolder, imageName)
        shutil.copy(path, destination)

# load all the image paths anh randomly shuffle them
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.DATA_PATH))
np.random.shuffle(imagePaths)

# generate training and validation paths
valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
trainPathLen = len(imagePaths) - valPathsLen
trainPaths = imagePaths[:trainPathLen]
valPaths = imagePaths[trainPathLen:]

# copy the training and validation images to their respective
copy_images(trainPaths, config.TRAIN)
copy_images(valPaths, config.VAL)