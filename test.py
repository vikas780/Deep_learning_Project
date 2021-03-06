"""
Deep learning project
"""

import numpy as np
import cv2
i=0

pic_num = 1

#input image
img=cv2.imread('text1.jpeg')

#copy of the orignal image is made
origImage=img.copy()

#display the original image
cv2.imshow('Input image',origImage)

#converting into gray scale image

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#apply OTSU Binarisation to the gray scale image 

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Finding contours to find the position of the dark pixels

cont, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

for d in cont:

    x,y,w,h=cv2.boundingRect(d)
    
    #drawing rectangular boxes for bounding the character
    cv2.rectangle(origImage,(x,y),(x+w,y+h),(0,255,0),1)

    #resultant segmented characters
    cv2.imshow('Character segmentation Input Image',origImage)

    
for d in cont:

    x,y,w,h=cv2.boundingRect(d)

    #finding the region of interest from the whole image
    roi = origImage[y:y+h, x:x+w]

    #each segmented character imagaes outputed to a image file  
    cv2.imwrite('Result/' + str(pic_num)+ '.jpg',roi)

    #increment variable pic_num to number the output images 
    pic_num = pic_num + 1


cv2.waitKey(0)
cv2.destroyAllWindows()
