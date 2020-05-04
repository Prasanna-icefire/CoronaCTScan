import cv2
import pickle
import os
import numpy as np 
import matplotlib.pyplot as plt 
import random
DATADIR = "/home/icefire/ML/CORONA/CoronaCTScan/Data_Images"
CATEGORIES = ['Negative','Positive']
training_data = []
IMG_SIZE = 60
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                newArray = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([newArray,class_num])
            except Exception as e:
                pass
create_training_data()            
print(len(training_data))
