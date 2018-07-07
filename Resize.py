import cv2
import os
from PIL import Image

def cropImage(img):
    resized_image = cv2.resize(img, (320, 240))
    return resized_image

def loadImages(folder):
    images = []
    dataFolder = 'TrainingData/'+folder
    for image in os.listdir(dataFolder):
        if any([image.endswith(x) for x in ['.jpeg','.pgm']]):
            img = cv2.imread(os.path.join(dataFolder,image))
            if img is not None:
                images.append(img)
    return images

def saveImage(img,i,folder):
    img = Image.fromarray(img)
    name = str(i)
    img.save('ProcessedData/'+folder+'/'+name+'.jpg')


folders = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
#folders = ['S1','S2','S3','S4']
for folder in folders:
    index = 1
    images = loadImages(folder)
    for image in images:
        img = cropImage(image)
        saveImage(img,index,folder)
        index = index+1

