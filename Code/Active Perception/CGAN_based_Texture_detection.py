#This is a simple code to test the model by using the offline image
# Pls go to line 79 and 89 to load the Picture and model
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
from tensorflow.keras.layers import *
from tensorflow.keras import *
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import sys
import math
from pathlib import Path
from Texture_Extraction import LocalBinaryPatterns
import pickle
from sklearn.svm import SVC

from scipy.interpolate import interp1d

def watersh(image):
    #Apply the watershed method
    #This function was implemented with following the tutorial in https://www.pyimagesearch.com/2015/11/02/watershed-opencv/

    #shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    #gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.threshold(gray, 0, 255,
	#cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5,5),np.uint8)

    image = cv2.erode(image,kernel,iterations = 1)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    D = ndimage.distance_transform_edt(image)
    localMax = peak_local_max(D, indices=False, min_distance=10,
        labels=image)

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=image)

    #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    centers = []
    masks = []
    max_area = 100

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255

        crop_cont, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crop_cont = crop_cont[0]

        area = int(cv2.contourArea(crop_cont))

        if area>max_area:
            max_area = area

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object

        x,y,w,h = cv2.boundingRect(c)
        #(x,y) = rect[0]
        #w = rect[1][0]
        #h = rect[1][1]
        #((x, y), radius) = cv2.minEnclosingCircle(c)

        centers.append((x,y,w,h,mask))


    return max_area,  centers

# read the image
image = cv2.imread(r'/content/tester.png')
depth = 20
#Prepare image for the model
inimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
inimg = tf.convert_to_tensor(inimg, dtype=tf.float32)
inimg = tf.image.resize(inimg, [480,640])
inimg = (inimg / 127.5) - 1
#------------------------------------------------------------------------------
# Defining the LBP extractor
desc = LocalBinaryPatterns(16, 4)
# Enumerations for class labels 
Fruits_labels_dict = {
    0: 'Apple',
    1: 'Strawberry',
    2: 'pepper' ,
    3: 'Tomato',
    4: 'Cherry' 
    
}
#Interpolation table data and interapolator
depth_cm = [10 , 15, 20 , 25 ,30]
picels_per_cm = [65 , 42.5 , 32.5 , 25 , 20]

polator = interp1d(depth_cm,picels_per_cm,kind='linear')
# Loading the texture classification model from disk
fnameWithPath1 = "/content/Texture_classification"
with open(fnameWithPath1, 'rb') as f1:
       rd = pickle.load(f1)
modelT = rd[0]
#--------------------------------------------------------------------------------
#load generator
generator = tf.saved_model.load(r"/content/Strawberry_Models")

#Pass the image through the model
inp = tf.expand_dims(inimg,0)
pred = generator(inp, training=True)[0]
pred = (pred *0.5 + 0.5) * 255
pred = pred.numpy()
pred = pred.astype(np.uint8)
pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
pred = cv2.GaussianBlur(pred, (5, 5), 1)
cv2.imwrite('output_of_GAN.jpg', pred)
#find blobs
image0 = cv2.resize(image, (1280,720))
temp_image = image0
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_org = cv2.resize(image, (1280,720))
image_seg_input = cv2.resize(pred, (1280,720))

image = cv2.cvtColor(image_seg_input, cv2.COLOR_BGR2GRAY)
ret, image = cv2.threshold(image,200,255,cv2.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((7,7),np.uint8)

#Morphological operations
image = cv2.erode(image,kernel2,iterations = 1)
image = cv2.dilate(image,kernel2,iterations = 2)
#image = cv2.erode(image,kernel2,iterations = 2)

image = cv2.bitwise_not(image)

#Finding Countours
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

kernel3 = np.ones((3,3),np.uint8)

max_area , centers= watersh(image)
count = 0
for (x,y,w,h,mask) in centers:
    crop_cont, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crop_cont = crop_cont[0]

    area = int(cv2.contourArea(crop_cont))

    if area < 1000: #max_area:
        continue
    else:
        count +=1
        perimeter = cv2.arcLength(crop_cont,True)
        k = cv2.isContourConvex(crop_cont)

        crop_cont_hux = cv2.convexHull(crop_cont)
        hux = np.zeros((720,1280,1), np.uint8)
        cv2.drawContours(hux, [crop_cont_hux], -1, (255,255,255), -1)
        

        #image0 = cv2.cvtColor(image_org, cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(image0, [crop_cont], -1, (0,255,0), 2)
        # Extracting the image region inside the bounding box
        crop_img = temp_image[int(y):int(y+h), int(x):int(x+w),:].copy()
        name = "crop"+str(count)+".jpg"
        # Resizing the region
        resz_img = cv2.resize(crop_img,(100,100),interpolation = cv2.INTER_AREA)
        cv2.imwrite(name,crop_img)
        # LBP calculation and label prediction
        gray = cv2.cvtColor(resz_img, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        hist = hist.reshape(1, -1)
        Tclass = modelT.predict(hist)
        Tlable = Fruits_labels_dict[Tclass[0]]
        # Physical Geometry calculation
        h_in_cm = h/polator(depth)
        w_in_cm = w/polator(depth)
        #cv2.drawContours(image0, [crop_cont_hux], -1, (0,255,0), 4)
        cv2.rectangle(image0,(x,y),(x+w,y+h),(0,255,0),2)
        # Adding tag to the image for displaying
        #tag = "h: "+str(h_in_cm)+" w: "+str(w_in_cm)
        tag = Tlable
        image0 = cv2.putText(image0, tag, (int(x+w),int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA, False)
        

# write the new image
cv2.imwrite('output.jpg', image0)
cv2.imwrite("huximg.jpg",hux)
