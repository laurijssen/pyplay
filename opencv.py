import getopt
import sys
import string
import os
import datetime
import random
import time
import io
import cv2
import zbar
import numpy
import imutils
import pygame
from matplotlib import pyplot as plt
from slackclient import SlackClient

from PIL import Image

def opencv():
    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    fullbody_cascade = cv2.CascadeClassifier('D:\\opencv\\build\\etc\\haarcascades\\haarcascade_fullbody.xml')
    face_cascade = cv2.CascadeClassifier('D:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('D:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

    #fullbody_cascade = cv2.HOGDescriptor()
    
    eye = cv2.imread('c:\\temp\\eye.png')

    ret, img = capture.read()
    #img = cv2.imread('c:\\temp\\face.jpg')
    #img = cv2.imread('c:\\temp\\body.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_barcode_data():
    cap = cv2.VideoCapture(0)
    scanner = zbar.ImageScanner()
    
    face_cascade = cv2.CascadeClassifier('D:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('D:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    scanner.parse_config('enable')

    l = ['c:\\temp\\barcode.jpg', 'c:\\temp\\qr.jpg']
        
    ret, img = cap.read()

    i = 0
    while True:
    #for s in l:
        #image = cv2.imread(s)
        ret, image = cap.read()
        #
        #frame = cv2.GaussianBlur(image, (0, 0), 3);
        #image = cv2.addWeighted(frame, 1.5, image, -0.5, 1.0, image)

        #cv2.imwrite('c:\\temp\\test.jpg', image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('img',image)
        cv2.waitKey(50)
        #cv2.destroyAllWindows()        

        #h,w = image.shape[:2]

        #image = Image.fromarray(gray)

        #stream = zbar.Image(w, h, 'Y800', image.tobytes())

        #scanner.scan(stream)

        #try:
        #    for sym in stream:
        #        print(sym.data + " " + str(i))            
        #        i += 1
        #except:
        #    pass

def motion_detection():
    #pygame.init()
    pygame.mixer.init()
    dogsound = pygame.mixer.Sound("c:\\temp\\dogs.wav")

    camera = cv2.VideoCapture(0)

    firstFrame = None

    while True:
        (grabbed, frame) = camera.read()
        text = 'Unoccupied'

        if not grabbed:
            break
        
        frame = imutils.resize(frame, width=500)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if firstFrame is None:
            firstFrame = gray
            continue

        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) < 4000:
                continue
            
            print(cv2.contourArea(c))

            (x,y,w,h) = cv2.boundingRect(c)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

            cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            #dogsound.play()
            time.sleep(0.5)

            cv2.imshow("Security Feed", frame)
            #cv2.imshow("Thresh", thresh)
            #cv2.imshow("Frame Delta", frameDelta)

            cv2.waitKey(100)

    camera.release()
    cv2.destroyAllWindows()


cv2.namedWindow('window')

img = None

def callback(i):
    global img
    res = img * (float(i) / 100.0)
    cv2.imshow('window', res)

def mouseCallbackDrawCircle(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        global img
        cv2.circle(img, (x, y), 115, (255,0, 0), 1)
        cv2.putText(img, "Tekst!!!", (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        cv2.imshow('window', img)

p1 = None
p2 = None
p2set = False

def mouseCallbackRegions(event, x, y, flags, userdata):
    global img, p2set
    global p1, p2

    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (x, y)        
        p2set = False
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        w, h = img.shape[:2]
        
        if x > w:
            x = w
        elif x < 0:
            x = 0

        if y > h:
            y = h
        elif y < 0:
            y = 0

        p2 = (x, y)
        p2set = True

        mask = numpy.ones(img.shape, dtype=numpy.bool)
        tmp = img * (mask.astype(img.dtype))
        
        cv2.rectangle(tmp, p1, p2, (0, 0, 255))
        w, h = tmp.shape[0:2]

        cv2.imshow('window', tmp)

    elif event == cv2.EVENT_LBUTTONUP and p2set:

        img2 = img[p1[1]:p2[1], p1[0]:p2[0]]

        hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        h += 10
        s += 10
        v += 10

        final_hsv = cv2.merge((h,s,v))
        img2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        #img2[:,:,2] += 255

        img[p1[1]:p2[1], p1[0]:p2[0]] = img2

        cv2.imshow('window', img)

def image_testing():
    global img
    img = cv2.imread('c:\\temp\\jungle.jpg', 1)

    if img == None:
        return

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #cv2.calcHist([img], [0], None, [256], [0, 1024])
    #plt.hist(img.ravel(), 256, [0, 256])
    #plt.show()
    
    #average_color_per_row = numpy.mean(img, axis=0)
    #average_color = numpy.mean(average_color_per_row, axis = 0)
    
    #print(int(round(average_color[0])))
    #print(int(round(average_color[1])))
    #print(int(round(average_color[2])))

    #img[:] = average_color

    #img = cv2.flip(img, 1)

    #res = numpy.float32(img)

    #cv2.createTrackbar('track', 'window', 0, 100, callback)
    cv2.setMouseCallback('window', mouseCallbackRegions)

    #cv2.imwrite('c:\\temp\\jungle2.jpg', img)

    cv2.imshow('window', img)
    cv2.waitKey()

def video_testing():
    cap = cv2.VideoCapture('c:\\temp\\film.avi')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
   
    while cap.isOpened():
        ret, frame = cap.read()
        if frame == None:
            break

        cv2.imshow('window', frame)

image_testing()
#video_testing()
cv2.waitKey(10000)

#motion_detection()
#detect_barcode_data()
#opencv()