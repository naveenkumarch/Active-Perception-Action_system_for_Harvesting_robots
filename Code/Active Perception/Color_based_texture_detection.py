from __future__ import division
import cv2
# to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin
#import pyzed.sl as sl
import logging
import time
from pathlib import Path
from Texture_Extraction import LocalBinaryPatterns
import pickle
from sklearn.svm import SVC

from scipy.interpolate import interp1d

green = (0, 255, 0)

YELLOW = [np.array(x, np.uint8) for x in \
            [[25,100,100], [35, 255, 255]] ]
    # hsv ranges for blue detections:
	#LBLUE - light blue
LBLUE = [np.array(x, np.uint8) for x in \
            [[100,100,100], [110, 255, 255]] ]
	#BLUE
BLUE = [np.array(x, np.uint8) for x in \
            [[115,100,100], [130, 255, 255]] ]
	#SBLUE - STRONGER BLUE
SBLUE = [np.array(x, np.uint8) for x in \
            [[130,100,100], [140, 255, 255]] ]
    # hsv ranges for red detections:
GENERALRED = [np.array(x, np.uint8) for x in \
            [[160,100,100], [180, 255, 255]] ]
    # red LEGOS
REDLEGOS = [np.array(x, np.uint8) for x in \
            [[165,100,100], [170, 255, 255]] ]
    # STRAWBERRY RED
RED = [np.array(x, np.uint8) for x in \
            [[0,100,100], [15, 255, 255]] ]
   # STRAWBERRY RED
OLDS = [np.array(x, np.uint8) for x in \
            [[20,100,100], [30, 255, 255]] ]

COLOR = REDLEGOS

COLOR2 = LBLUE
# number of blobs to detect
MAX_DETECTIONS =4 
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
#-------------------------------------------------------------------------------


def get_filtered_contours(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image1 = image.copy()
        # convert image to color space hsv
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # threshold the image to remove all colors that are not the one selected
    frame_threshed = cv2.inRange(hsv_img, COLOR[0], COLOR[1])
    ret,thresh = cv2.threshold(frame_threshed, COLOR[0][0], 255, 0)

        # create the list of filtered contours to return
    filtered_contours = []

        # find all contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # sort all contours by area, largest to smallest
    contour_area = [ (cv2.contourArea(c), (c) ) for c in contours]
    contour_area1 = sorted(contour_area,reverse=True, key=lambda x: x[0])


    for j, (area,(cnt)) in enumerate(contour_area1):
            # only report MAX_DETECTIONS number of controus
        if j >=MAX_DETECTIONS: break

            # create a bounding box around the contour
	    # x,y,w,h are the four points for the contour making
	    # the corners of the square and the bounding rectangle
	    # of them is created below
        x,y,w,h = cv2.boundingRect(cnt)
        box = (x,y,w,h)
        Cx = (w/2)+x
        Cy = (h/2)+y
        
	    #text_file = open("Output.txt", "w+")
	        #text_file = open("Output.txt", "a+")
	    #text_file.write(str(Cx)+" "+str(Cy)+'\n')
	    #text_file.close


            # make the color of the box the mean of the contour
        mask = np.zeros(thresh.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        mean_val = cv2.mean(img,mask = mask)
            # add this contour to the list
        filtered_contours.append((cnt, box, mean_val) )


    return filtered_contours, image1




# read the image
image = cv2.imread(r'/content/Explorer_HD720_SN10028623_10-48-27.png')
temp_image = image.copy()
# detect it
big_blue_contour, image1 = get_filtered_contours(image)
bgr = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
count = 0
for (cnt, box, mean_val) in big_blue_contour:
    x,y,w,h=cv2.boundingRect(cnt)
    box = (x,y,w,h)
    Cx = (w/2)+x
    Cy = (h/2)+y
    
    startX = int(Cx-50)
    startY = int(Cy-50)
    endX = int(Cx+50)
    endY = int(Cy+50)
    bounds = tuple([Cx,Cy,w,h])
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
    #h_in_cm = h/polator(depth)
    #w_in_cm = w/polator(depth)
    count+=1
    #cv2.drawContours(image0, [crop_cont_hux], -1, (0,255,0), 4)
    #cv2.rectangle(image0,(x,y),(x+w,y+h),(0,255,0),2)
    #tag = "h: "+str(h_in_cm)+" w: "+str(w_in_cm)
    tag = str(Tlable)
    bgr = cv2.putText(bgr, tag, (int(x+w),int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA, False)
    #x, y, z = get_object_depth(depth, bounds)
    cv2.rectangle(bgr, (startX, startY), (endX, endY), (0,255,0),2)

# write the new image
cv2.imwrite('yo2.jpg', bgr)
