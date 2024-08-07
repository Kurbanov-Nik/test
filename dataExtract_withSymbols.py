import os
import math
import cv2
import numpy as np
import pandas as pd

gt = pd.DataFrame(columns=['file_name', 'value'])

filePath = "ru_sum_num"
corePath = "out_2/imgs/" # "meshups/"

with open("./%s%s/labels.txt" % (corePath, filePath), 'r', encoding="utf-8") as f:
    for index, row in enumerate(f):
        print(row)
        if row[0] == '#':
            continue
        row = row.split(".jpg ")
        gt.loc[gt.shape[0]] = (row[0], row[-1].replace("\n", ""))

print(gt)

def preprocess(img, shapeTo = (64, 800)):
    if img.shape[0] > shapeTo[0] or img.shape[1] > shapeTo[1]:
        shrinkMultiplayer = min(math.floor(shapeTo[0] / img.shape[0] * 100) / 100,
                                 math.floor(shapeTo[1] / img.shape[1] * 100) / 100)
        img = cv2.resize(img, None,
                         fx=shrinkMultiplayer,
                         fy=shrinkMultiplayer,
                         interpolation=cv2.INTER_AREA)

    img = cv2.copyMakeBorder(img, math.ceil(shapeTo[0] / 2) - math.ceil(img.shape[0] / 2),
                             math.floor(shapeTo[0] / 2) - math.floor(img.shape[0] / 2),
                             math.ceil(shapeTo[1] / 2) - math.ceil(img.shape[1] / 2),
                             math.floor(shapeTo[1] / 2) - math.floor(img.shape[1] / 2),
                             cv2.BORDER_CONSTANT, value=255)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                 21, 10).astype('bool')

train_X, train_y = [], []

for folder in os.listdir("./%s%s" % (corePath, filePath)):
    if folder != "labels.txt":
        label = (gt[gt.file_name == folder[:folder.rfind('.')]].value.tolist())[0]
        train_X.append(preprocess(cv2.imread("./%s%s/%s" % (corePath, filePath, folder), 0)))
        train_y.append(label)

train_X = np.array(train_X)
np.save("./out_2/prepared_data/%s_data.npy" % (filePath), train_X)
np.save("./out_2/prepared_data/%s_labels.npy" % (filePath), np.array(train_y))
