import cv2
import numpy
import time

import sys
sys.path.append('../')

from utils import save_and_show

# Using HOG with OpenCV:

# Need to initialize the detector and specify that you want to use the detector for
# pedestrians

hog = cv2.HOGDescriptor() # initialize detector
det = cv2.HOGDescriptor_getDefaultPeopleDetector() # use detector for pedestrians
hog.setSVMDetector(det) # set to people

def HOG_detect_pedestrians(file, stride, scale, hitThreshold, finalThreshold):
    start = time.time()
    original_img = cv2.imread(file)
    hog_img = original_img.copy()

    # call detectMultiScale()
    (boxes, weights) = hog.detectMultiScale(hog_img, winStride=(stride, stride), padding=(0,0), scale=scale, hitThreshold=hitThreshold, finalThreshold=finalThreshold)

    # highlight pedestrians with boxes
    for i, (x, y, w, h) in enumerate(boxes):
        # ~ cv2.putText(image, 'Text', (x, y), cv2.FONT_HERSHEY_PLAIN, 2, clr)
        cv2.putText(hog_img, '%.2f' % weights[i], (x, y + h - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255)) 
        # ~ '%.2f' % weights[i] says describe weights[i] as a float to the nearest 2 decimal points 

        # ~ cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2m,)
        cv2.Rectangle(hog_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # ~ cv2.imwrite("out.jpg", image)
    cv2.imwrite("hog_img_out" + str(stride) + "_" + str(scale) + "_" + str(hitThreshold) + "_" str(finalThreshold)) 