# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:25:23 2021

@author: chand
"""

# Echo client program
import socket
import time
import struct
from datetime import datetime
import pandas ad pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import imutils
import numpy as np
import math
import statistics

from skimage.segmentation import watershed
from tensorflow.keras.layers import *
from tensorflow.keras import *
import logging
import pyzed.sl as sl #ZED API：https://github.com/stereolabs/zed-python-api
from skimage.feature import peak_local_max
from scipy import ndimage

writer = csv.writer(f)
Headers = ['X_arm','Y_arm','Z_arm','X_cam','Y_cam','Z_cam']

# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HOST = "192.168.0.9" # The remote host
PORT_30003 = 30003
PORT_30002 = 30002

print ("Starting Program")

row_count = 0
home_status = 0
program_run = 0

Angles_data = pd.read_csv("D:\MSC\Cognitive Robotics\Scripts\Angles_in_radinas.txt", sep=",", squeeze=False)

j_data = np.array(Angles_data)
shape = np.shape(j_data)
rows = shape[0]

def   (depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.
    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median

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

        rect = cv2.minAreaRect(c)
        (x,y) = rect[0]
        w = rect[1][0]
        h = rect[1][1]
        #((x, y), radius) = cv2.minEnclosingCircle(c)

        centers.append((x,y,w,h,mask))


    return max_area,  centers
    
def main():
    # Launch camera by id
    zed_id=0
    input_type = sl.InputType()
    input_type.set_from_camera_id(zed_id)
    init = sl.InitParameters(input_t=input_type)
    init.coordinate_units = sl.UNIT.METER

    cam = sl.Camera()
    if not cam.is_opened():
        log.info("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    # Use STANDARD sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()

    #load generator
    generator = tf.saved_model.load(r"C:\Users\fuliw\Documents\python\Pix2pix\Strawberry_Models")

    log.info("Running...")

    key = ''
    while key != 113:  # for 'q' key
        start_time = time.time() # start time of the loop
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            if (row_count<rows):
                if program_run == 0:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.settimeout(10)
                        s.connect((HOST, PORT_30002))
                        time.sleep(0.05)
                        s.send ("set_digital_out(2,True)" + "\n")
                        time.sleep(0.1)
                        print ("0.2 seconds are up already")
                        s.send ("set_digital_out(7,True)" + "\n")
                        time.sleep(2)
                        command = "movej( ["+str(arr_dat[row_count,0])+' , '+str(arr_dat[row_count,1])+" , "+str(arr_dat[row_count,2])+' , '+str(arr_dat[row_count,3])+str(arr_dat[row_count,4])+' , '+str(arr_dat[row_count,5])+ "], a=1.3962634015954636, v=1.0471975511965976)"
                        s.send(command+"\n")
                        s.close()
                        s.connect((HOST, PORT_30003))
                        time.sleep(5.0)
                        print ("")
                        packet_1 = s.recv(4)
                        packet_2 = s.recv(8)
                        packet_3 = s.recv(48)
                        packet_4 = s.recv(48)
                        packet_5 = s.recv(48)
                        packet_6 = s.recv(48)
                        packet_7 = s.recv(48)
            
                        packet_8_1 = s.recv(8)
                        packet_8_1 = packet_8_1.encode("hex")
                        joint_1 = str(packet_8_1)
                        joint_1 = struct.unpack('!d',packet_8_1.decode('hex'))[0]
                        print("joint 1 pos:",joint_1)
            
                        packet_8_2 = s.recv(8)
                        packet_8_2 = packet_8_2.encode("hex")
                        joint_2 = str(packet_8_2)
                        joint_2 = struct.unpack('id',packet_8_2.decode('hex'))[0]
                        print("joint 2 pos:",joint_2)
            
                        packet_8_3 = s.recv(8)
                        packet_8_3 = packet_8_3.encode("hex")
                        joint_3 = str(packet_8_3)
                        joint_3 = struct.unpack('id',packet_8_3.decode('hex'))[0]
                        print("joint 3 pos:",joint_3)
            
                        packet_8_4 = s.recv(8)
                        packet_8_4 = packet_8_4.encode("hex")
                        joint_4 = str(packet_8_4)
                        joint_4 = struct.unpack('id',packet_8_4.decode('hex'))[0]
                        print("joint 4 pos:",joint_4)
            
                        packet_8_5 = s.recv(8)
                        packet_8_5 = packet_8_5.encode("hex")
                        joint_5 = str(packet_8_5)
                        joint_5 = struct.unpack('id',packet_8_5.decode('hex'))[0]
                        print("joint 5 pos:",joint_5)
            
                        packet_8_6 = s.recv(8)
                        packet_8_6 = packet_8_6.encode("hex")
                        joint_6 = str(packet_8_6)
                        joint_6 = struct.unpack('id',packet_8_6.decode('hex'))[0]
                        print("joint 6 pos:",joint_6)
            
                        packet_9 = s.recv(48)
                        packet_10 = s.recv(48)
                        packet_11 = s.recv(48)

                        packet_12 = s.recv(8)
                        packet_12 = packet_12.encode("hex") #convert the data from \x hex notation to plain hex
                        x = str(packet_12)
                        x = struct.unpack('!d', packet_12.decode('hex'))[0]
                        print ("X = ", x * 1000)

                        packet_13 = s.recv(8)
                        packet_13 = packet_13.encode("hex") #convert the data from \x hex notation to plain hex
                        y = str(packet_13)
                        y = struct.unpack('!d', packet_13.decode('hex'))[0]
                        print ("Y = ", y * 1000)

                        packet_14 = s.recv(8)
                        packet_14 = packet_14.encode("hex") #convert the data from \x hex notation to plain hex
                        z = str(packet_14)
                        z = struct.unpack('!d', packet_14.decode('hex'))[0]
                        print ("Z = ", z * 1000)

                        packet_15 = s.recv(8)
                        packet_15 = packet_15.encode("hex") #convert the data from \x hex notation to plain hex
                        Rx = str(packet_15)
                        Rx = struct.unpack('!d', packet_15.decode('hex'))[0]
                        print ("Rx = ", Rx)

                        packet_16 = s.recv(8)
                        packet_16 = packet_16.encode("hex") #convert the data from \x hex notation to plain hex
                        Ry = str(packet_16)
                        Ry = struct.unpack('!d', packet_16.decode('hex'))[0]
                        print ("Ry = ", Ry)

                        packet_17 = s.recv(8)
                        packet_17 = packet_17.encode("hex") #convert the data from \x hex notation to plain hex
                        Rz = str(packet_17)
                        Rz = struct.unpack('!d', packet_17.decode('hex'))[0]
                        print ("Rz = ", Rz)
                        now = datetime.now()

                        current_time = now.strftime("%H:%M:%S")
                        print("Current Time =", current_time)

                        home_status = 1
                        program_run = 0
                        s.close()
                    except socket.error as socketerror:
                        print("Error: ", socketerror)
                        print ("Program finish")
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            image = mat.get_data() #image from ZED camera

            cam.retrieve_measure(
                point_cloud_mat, sl.MEASURE.XYZRGBA)
            depth = point_cloud_mat.get_data() #depth information

            #Prepare image for the model
            inimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inimg = tf.convert_to_tensor(inimg, dtype=tf.float32)
            #image.ORGSIZE_H = inimg.shape[0]
            #image.ORGSIZE_W = inimg.shape[1]

            inimg = tf.image.resize(inimg, [480,640])
            inimg = (inimg / 127.5) - 1

            #Pass the image through the model
            inp = tf.expand_dims(inimg,0)
            pred = generator(inp, training=True)[0]
            pred = (pred *0.5 + 0.5) * 255
            pred = pred.numpy()
            pred = pred.astype(np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)


            #bounds = watersh(pred)
            image0 = cv2.resize(image, (1280,720))
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
            max_area , centers= watersh(image)
                #centers = [(cX,cY,deep)]

                # Do the detection
            for (x,y,w,h,mask) in centers:
                crop_cont, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                crop_cont = crop_cont[0]


                area = int(cv2.contourArea(crop_cont))
                bound=tuple([x,y,w,h])

                Cx, Cy, Cz = get_object_depth(depth, bound)
                
                acceptable_area = 1000/(Cz*5)
                if area < acceptable_area:
                    continue
                else:

                    crop_cont_hux = cv2.convexHull(crop_cont)

                    bound=tuple([x,y,w,h])

                    Cx, Cy, Cz = get_object_depth(depth, bound)
                    #distance = math.sqrt(Cx * Cx + Cy * Cy + Cz * Cz)
                    print('Locations:')
                    print(Cx, Cy, Cz)
                    cv2.drawContours(image0, [crop_cont_hux], -1, (0,255,0), 4)
                    temp = [x,y,z,Cx,Cy,Cz]
                    writer.writerow(temp)

            cv2.imshow("ZED", image0)
            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()
    log.info("\nFINISH")

if __name__ == "__main__":
    main()
