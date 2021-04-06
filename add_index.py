import cv2
import numpy as np
import imutils
from tqdm import tqdm
import face_recognition
from imutils import paths
import matplotlib.pyplot as plt
import os
from mtcnn.mtcnn import MTCNN
import hnswlib
from constant import DIM, NUM_ELEMENTS, IMAGE_SIZE, EF_CONSTRUCTION, DIMENTIONAL, M
import pandas as pd
from pandas import DataFrame
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
def add(name):

    p = hnswlib.Index(space='l2', dim=DIM)
    p.load_index("images.bin", max_elements = NUM_ELEMENTS)
    imagePaths = list(paths.list_images('images'))
    detector = MTCNN()

    def check_image_path(imagePath):
        img = face_recognition.load_image_file(imagePath)
        try:
            coodirnate = detector.detect_faces(img)[0]['box']
        except IndexError:
            coodirnate = []
        return coodirnate, img

    def image_encoding(coodirnate, img):
        x, y, w, h = [v for v in coodirnate]
        x2, y2 = x + w, y + h
        face = img[y:y+h, x:x+w]
        img_emb = face_recognition.face_encodings(face)
        return img_emb

    # print(imagePaths)
    imp = []
    C = pd.read_csv('Index.csv')
    x = list(C['name'])
    b = []
    # name = ''
    for i in imagePaths:
        if name in i:
            imp.append(i)
    for i in tqdm(range(len(imp))):
        coodirnate, img = check_image_path(imp[i])
        
        if img.shape == (IMAGE_SIZE, IMAGE_SIZE, DIMENTIONAL):
            img_emb  = face_recognition.face_encodings(img)
            if len(img_emb) == 0:
                pass
            else:
                p.add_items(np.expand_dims(img_emb[0], axis = 0), len(x) + i)
        else:    
            if len(coodirnate) == 0:
                pass
            else:
                img_emb = image_encoding(tuple(coodirnate), img)
                if len(img_emb) == 0:
                    pass
                else:
                    p.add_items(np.expand_dims(img_emb[0], axis = 0), len(x) + i)
            
        b.append(name)

    x = x + b
    C = pd.Series(x)
    df = DataFrame(C, columns= ['name'])
    export_csv = df.to_csv ('Index.csv', index = None, header=True) # here you have to write path, where result file will be stored

    index_path='images.bin'
    print("Saving index to '%s'" % index_path)
    p.save_index("images.bin")
    del p