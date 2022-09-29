# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:25:23 2021

@author: chandra Naveen
"""
#-------------------------------------------------------------------------------------------------
# Echo client program
import socket
import time
import struct
from datetime import datetime
import csv
#-------------------------------------------------------------------------------------------------

"""
IP address of the robotic arm to connect and the TCP IP port numbers for reading information from the arm 
"""
HOST = "192.168.0.9" # The remote host
PORT_30003 = 30003
PORT_30002 = 30002

print ("Starting Program")

count = 1
home_status = 0
program_run = 0

"""
Opening a csv File in append mode to collect data related to joint angles along with time stamp of the machine.
"""
f = open("Joint_angles_data.csv", mode='a', encoding='utf-8', newline='')
writer = csv.writer(f)
Headers = ['joint1','joint2','joint3','joint4','joint5','joint6','X','Y','Z','Rx','Ry','Rz','time_stamp']

#----------------------------------------------------------------------------
#Please comment out the below line if not running the file for first line.
writer.writerow(Headers)
#----------------------------------------------------------------------------


while (True):
    
    print ("")
    if program_run == 0:
        try:
            #connecting to the scoket for message reception
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(4.5)
            s.connect((HOST, PORT_30003))
            time.sleep(0.5)
            
            # Recieving and braking down the signal 
            packet_1 = s.recv(4)
            packet_2 = s.recv(8)
            packet_3 = s.recv(48)
            packet_4 = s.recv(48)
            packet_5 = s.recv(48)
            packet_6 = s.recv(48)
            packet_7 = s.recv(48)
            
            # Joint angle data reception
            packet_8_1 = s.recv(8)
            packet_8_1 = packet_8_1.encode("hex")
            joint_1 = str(packet_8_1)
            joint_1 = struct.unpack('!d',packet_8_1.decode('hex'))[0]
            #print("joint 1 pos:",joint_1)
            
            packet_8_2 = s.recv(8)
            packet_8_2 = packet_8_2.encode("hex")
            joint_2 = str(packet_8_2)
            joint_2 = struct.unpack('!d',packet_8_2.decode('hex'))[0]
            #print("joint 2 pos:",joint_2)
            
            packet_8_3 = s.recv(8)
            packet_8_3 = packet_8_3.encode("hex")
            joint_3 = str(packet_8_3)
            joint_3 = struct.unpack('!d',packet_8_3.decode('hex'))[0]
            #print("joint 3 pos:",joint_3)
            
            packet_8_4 = s.recv(8)
            packet_8_4 = packet_8_4.encode("hex")
            joint_4 = str(packet_8_4)
            joint_4 = struct.unpack('!d',packet_8_4.decode('hex'))[0]
            #print("joint 4 pos:",joint_4)
            
            packet_8_5 = s.recv(8)
            packet_8_5 = packet_8_5.encode("hex")
            joint_5 = str(packet_8_5)
            joint_5 = struct.unpack('!d',packet_8_5.decode('hex'))[0]
            #print("joint 5 pos:",joint_5)
            
            packet_8_6 = s.recv(8)
            packet_8_6 = packet_8_6.encode("hex")
            joint_6 = str(packet_8_6)
            joint_6 = struct.unpack('!d',packet_8_6.decode('hex'))[0]
            #print("joint 6 pos:",joint_6)
            
            packet_9 = s.recv(48)
            packet_10 = s.recv(48)
            packet_11 = s.recv(48)

            packet_12 = s.recv(8)
            packet_12 = packet_12.encode("hex") #convert the data from \x hex notation to plain hex
            x = str(packet_12)
            x = struct.unpack('!d', packet_12.decode('hex'))[0]
            #print ("X = ", x * 1000)

            packet_13 = s.recv(8)
            packet_13 = packet_13.encode("hex") #convert the data from \x hex notation to plain hex
            y = str(packet_13)
            y = struct.unpack('!d', packet_13.decode('hex'))[0]
            #print ("Y = ", y * 1000)

            packet_14 = s.recv(8)
            packet_14 = packet_14.encode("hex") #convert the data from \x hex notation to plain hex
            z = str(packet_14)
            z = struct.unpack('!d', packet_14.decode('hex'))[0]
            #print ("Z = ", z * 1000)

            packet_15 = s.recv(8)
            packet_15 = packet_15.encode("hex") #convert the data from \x hex notation to plain hex
            Rx = str(packet_15)
            Rx = struct.unpack('!d', packet_15.decode('hex'))[0]
            #print ("Rx = ", Rx)

            packet_16 = s.recv(8)
            packet_16 = packet_16.encode("hex") #convert the data from \x hex notation to plain hex
            Ry = str(packet_16)
            Ry = struct.unpack('!d', packet_16.decode('hex'))[0]
            #print ("Ry = ", Ry)

            packet_17 = s.recv(8)
            packet_17 = packet_17.encode("hex") #convert the data from \x hex notation to plain hex
            Rz = str(packet_17)
            Rz = struct.unpack('!d', packet_17.decode('hex'))[0]
            #print ("Rz = ", Rz)
            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            #print("Current Time =", current_time)
            temp = [joint_1, joint_2, joint_3. joint_4, joint_5, joint_6, x, y, z, Rx, Ry, Rz, str(current_time)]
            writer.writerow(temp)
            home_status = 1
            program_run = 0
            s.close()
        except socket.error as socketerror:
            print("Error: ", socketerror)
            print ("Program finish")
            f.close()