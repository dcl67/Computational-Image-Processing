#!/usr/bin/env python
import sys

import numpy as np
import cv2

def logImpl(img):
    # return the log of the image
    # Since log(0) is undefined, add 1 before performing the operation.
    # Be sure to normalize such that the maximum value is 1
    return np.log1p(img.max())
    
def powerLaw(img, c, y):
    plaw = c*img**(-1*y)
	
    return plaw

def process(image, operation, args):
    
    inputChannel = np.array(b,g,r)
    # First, determine the input channel.
    # This appears to be split into blue, green, red channels through opencv split
    inputChannel = cv2.split(image)
	
    # If the image has 3 channels, assume it is BGR, convert it to HSV, and operate on the value channel.
    # Use http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html to guide you here.
    if(len(image.shape) != 3):
        hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
        hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
        hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
        # If the image has 1 channel, just operate on it directly
    elif(len(image.shape) == 1):
	    hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    else:
        print("idk bro. skipping whatever you're on")
        pass

    # Convert the inputChannel to be 32-bit floating point representation, and scale it such
    # that the values are between 0 and 1
	
    
    print "input from %f to %f " % (np.min(inputChannel.flatten()), np.max(inputChannel.flatten()))
    
    if (operation == "log"):
        # For the log operation, implement and call logImpl above
        logImpl(image)
        
    elif (operation == "powerlaw"):
        # Parse c and y from the argument list, and convert them to floats
        cfloat = sys.argv[0]
        yfloat = sys.argv[1]
        powerLaw(image,cfloat,yfloat) #
        # Call powerLaw
        
    elif (operation == "ilog"):
        # the inverse log is exp(img) normalized to be max 1 (hint: divide by the max possible value)
        print('ilog')
        
    elif (operation == "negative"):
        # What is the negative? Should be easy to code :D
        image*-1
        
    elif (operation == "npow"):
        # Parse n from the argument list and convert it to a float.
        print('npow')
        # compute inputChannel to the power of x. 
        
    elif (operation == "nroot"):
        # Nth rooth is the same as npow(1.0/n)
        print('nroot')
    else:
        raise Exception('Unknown operation {}'.format(operation))
        
    print "result from %f to %f " % (np.min(resultChannel.flatten()), np.max(resultChannel.flatten()))
        
        
    # Convert the resultChannel to be 8-bit unsigned representation, and scale it such
    # that the values are between 0 and 255
    
    
    # Construct the result image
    
        # If the image had 3 channels, replace the V channel of the HSV image with the result channel,
        # and convert it back to BGR
        
        # If the image has 1 channel, just return it
        
    
    return resultImage

if __name__ == "__main__":
    try:
        inputFileName, outputFileName, operation = sys.argv[1:4]
    except:
        print("Usage: python hw0.py inputFile outputFile operation ")
        exit(1)

    image = cv2.imread(inputFileName)
    result = process(image, operation, sys.argv[4:])
    cv2.imwrite(outputFileName, result)
