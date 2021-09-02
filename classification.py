import numpy as np
import mahotas
import cv2
import os
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.feature import corner_harris, corner_subpix, corner_peaks, peak
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from skimage.morphology import skeletonize

# location holding the directories with images 
file_path = r'/home/nafiseh/Pictures/flowers/flower'

# create empty lists to hold the images being read

x_feature = []
y_label = []

labels = os.listdir(file_path) 


def corner(img):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray)
    corners = cv2.goodFeaturesToTrack(image_gray, 10, 0.04, 11)
    corners = np.int0(corners)
    re_corners = np.reshape(corners, (10, 2))
    return re_corners.flatten()

def skeleton(img):
    skeleton = skeletonize(invert(img))
    return skeleton.flatten()

def hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick     

def histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

fixed_size = (400, 400)

for dirname in labels:
    filepath = os.path.join(file_path, dirname)
    for file in os.listdir(filepath):
        filename = os.path.join(filepath, file)
        # print(filename)
        image = cv2.imread(filename)
        image_resized = cv2.resize(image, fixed_size)
        
        corners = corner(image_resized)
        skelet = skeleton(image_resized)

        #global features
        hu = hu_moments(image_resized)
        har = haralick(image_resized)
        his = histogram(image_resized)

        f_vector_local = np.hstack([corners, skelet])
        f_vector_global = np.hstack([hu, har, his])
        
        x_feature.append(f_vector_global)
        y_label.append(dirname)
        continue
    continue
    break



# convert lists to numpy array
x_image_n = np.array(x_feature)
y_label_n = np.array(y_label)


#test and train data
x_train, x_test, y_train, y_test = train_test_split(x_image_n, y_label_n, test_size=0.2, random_state=0)

clf_rfc = RandomForestClassifier().fit(x_train, y_train)
y_pred_rfc = clf_rfc.predict(x_test)

print(precision_score(y_test, y_pred_rfc, average='weighted'))
print(accuracy_score(y_test, y_pred_rfc))  
print(confusion_matrix(y_test, y_pred_rfc))

