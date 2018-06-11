#!/usr/bin/env python

"""tonemap -- Tone map an HDR image.

Tonemap an HDR input image using a global method or local method.
"""

import argparse
import sys
import cv2
import numpy as np
import scipy.misc

import Imath
import OpenEXR


def loadexr(filename):
    """Open an exr image and return a numpy array."""
    f = OpenEXR.InputFile(filename)
    dw = f.header()['dataWindow']
    sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    image = np.empty((sz[0], sz[1], 3), dtype=np.float32)
    image[..., 0] = np.fromstring(f.channel("R", FLOAT), dtype=np.float32).reshape(sz)
    image[..., 1] = np.fromstring(f.channel("G", FLOAT), dtype=np.float32).reshape(sz)
    image[..., 2] = np.fromstring(f.channel("B", FLOAT), dtype=np.float32).reshape(sz)

    return image



def bilateral_tonemap(image, debug, sigmaRange, sigmaSpace, contrast, gamma):
    """Tonemap image (HDR) using Durand 2002"""
    # make sure we avoid numerical issues.
    imgMin = np.min(image.flatten())
    if imgMin < 0:
        image = image - imgMin
    
    # Compute intensity as the average of the three color channels.
    intensity = np.mean(image, axis=2)
    
    # compute log intensity
    logintensity = np.log(intensity) #unless we have to do this per-channel
    
    # Make sigma space a factor of the image resolution
    width, height = image.shape[0:2]
    sigmaSpace = sigmaSpace * min(width,height)
    
    # apply the bilateral filter to the result. use a kernel size of 14 and the provided sigmaRange sigmaSpace
    # You can use opencv here if you like.
    baseImage = cv2.bilateralFilter(logintensity, 14, sigmaRange, sigmaSpace)
    
    # compute detail image
    detail = logintensity - baseImage 
    
    # Compute the min and max values of the base image 
    max_value = np.max(baseImage)
    min_value = np.min(baseImage)
    
    # Reduce the dynamic range of the base image as discussed in the slides.
    # note that dR = log(contrast)
    contrastReducedBaseImage = (baseImage - max_value)*(np.log(contrast)/(max_value - min_value))

    # Reconstruct the intensity from the adjusted base image and the detail image.
    reconst = np.exp(contrastReducedBaseImage + detail)
    
    # Put the colors back in.
    b,g,r = cv2.split(image)
    b_prime = reconst * (b/intensity)
    g_prime = reconst * (g/intensity)
    r_prime = reconst * (r/intensity)
    colorReconstructedImage = cv2.merge((b_prime,g_prime,r_prime))

    # Apply any gamma correction requested
    b_primer = b_prime**gamma
    g_primer = g_prime**gamma
    r_primer = r_prime**gamma
    gammaCorrected = cv2.merge((b_primer,g_primer,r_primer))
    
    # Normalize the results. Do each channel separately.
    for c in range(0,3):
        gammaCorrected[:,:,c] = (gammaCorrected[:,:,c] - np.min(gammaCorrected[:,:,c]))/(np.max(gammaCorrected[:,:,c])-np.min(gammaCorrected[:,:,c]))

    # Write the debug info if requested.
    if (debug):
        print "intensity range ", np.min(intensity), np.max(intensity)
        print "logintensity range ", np.min(logintensity), np.max(logintensity)
        print "baseImage range ", np.min(baseImage), np.max(baseImage)
        print "detail range ", np.min(detail), np.max(detail)
        print "contrastReducedBaseImage range ", np.min(contrastReducedBaseImage), np.max(contrastReducedBaseImage)
        for c in range(0, 3):
            print "gammaCorrected " + str(c) + " range ", np.min(gammaCorrected[:,:,c]), np.max(gammaCorrected[:,:,c])
    
        cv2.imwrite("debug-intensity.png", intensity * 255)
        cv2.imwrite("debug-logintensity.png", logintensity * 255)
        cv2.imwrite("debug-baseImage.png", np.exp(baseImage) * 255)
        cv2.imwrite("debug-detail.png", detail * 255)
        cv2.imwrite("debug-contrastReducedBaseImage.png", np.exp(contrastReducedBaseImage) * 255)
        cv2.imwrite("debug-reconst.png", reconst * 255)
        scipy.misc.imsave("debug-colorReconstructedImage.png", colorReconstructedImage)
    
    # Convert to 8-bit and return
    return (gammaCorrected *  255).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--debug', action='store_true',
                        help="dump debug data")
    parser.add_argument('--sigmarange', type=float, default=0.4,
                        help="The range (intensity) variance for the bilateral filter")
    parser.add_argument('--sigmaspace', type=float, default=0.05,
                        help="The space variance for the bilateral filter")
    parser.add_argument('--contrast', type=float, default=50.,
                        help="A constance specifying how much contrast to produce. Values between 5 and 200 are acceptable.")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="Gamma corrects the reconstructed image by this factor.")
    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()

    scipy.misc.imsave(args.output_file, bilateral_tonemap(loadexr(args.input_file), args.debug, args.sigmarange, args.sigmaspace, args.contrast, args.gamma))
