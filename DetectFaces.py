import cv2
from PIL import Image
import time
import StringIO
import glob
import numpy as np
import os

TOTAL_FACES = 0
START = time.time()

def convertToRgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def saveImage(img):
    global PIC, FOLDER
    img = Image.fromarray(img)
    img.save("TrainingData/" + FOLDER + "/" + str(PIC) + ".jpg")
    PIC = PIC + 1


def cropImage(img, rect):
    (x, y, w, h) = rect
    i = img[y:y + h, x:x + w]
    i = cv2.resize(i, (60, 60))
    return i


def detectFaces(img):
    global TOTAL_FACES
    face_cascade = cv2.CascadeClassifier('Cascades/lbpcascade_profileface.xml')
    face_cascade1 = cv2.CascadeClassifier('Cascades/haarcascade_frontalface.xml')
    eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
    face_cascade2 = cv2.CascadeClassifier('Cascades/Face_cascade.xml')
    faces = []
    faces = face_cascade1.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2, minSize=(70, 70))
    global eye
    if len(faces) == 0:
        print("Inside profile")
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(70, 70))
        if len(faces) == 0:
            print("Inside flip")
            img = cv2.flip(img, 1)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(70, 70))
    print(faces)
    return faces


def loadImages(dataFolder):
    images = []
    for image in os.listdir(dataFolder):
        if any([image.endswith(x) for x in ['.png', '.jpg']]):
            img = cv2.imread(os.path.join(dataFolder, image))
            if img is not None:
                images.append(image)
    return images


def extractFaces():
    global PIC, FOLDER, START, TOTAL_FACES
    folders = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    print("Extracting Faces.....")
    for folder in folders:
        PIC = 1
        FOLDER = folder
        index = 0
        dataFolder = 'ProcessedData/' + folder
        images = loadImages(dataFolder)
        for image in images:
            imagePath = dataFolder + "/" + image
            print(imagePath)
            imageArray = cv2.imread(imagePath)
            faces = detectFaces(imageArray)
            if len(faces) > 0:
                TOTAL_FACES = TOTAL_FACES + 1
                croppedImage = cropImage(imageArray, faces[0])
                saveImage(croppedImage)
                index = index+1
            else:
                imageArray = histogram_equalize(imageArray)
                faces = detectFaces(imageArray)
                if len(faces) > 0:
                    TOTAL_FACES = TOTAL_FACES + 1
                    croppedImage = cropImage(imageArray, faces[0])
                    saveImage(croppedImage)
                    index = index+1
            if index==568:
                break
    timeTaken = time.time() - START
    timeTaken = timeTaken / 60
    print("Extracting Faces took: %.2f mins" % round(timeTaken, 2))
    print("Total Faces: " + str(TOTAL_FACES))
    print("Accuracy: %.2f %%" % round(((TOTAL_FACES / 5850) * 100), 2))


