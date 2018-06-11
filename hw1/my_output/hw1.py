#!/usr/bin/env python
import sys

import numpy as np
import cv2

def remove_vertical_seam(image, seam):
    """
    Removes the given seam from the image.
    
    image : an n x m array (may have multiple channels)
    seam : an n x 1 array of X-coordinates defining the seam pixels in top-down order.
    
    Thus, seam[0] means remove pixel (0, seam[0]) from the input image.
    
    returns: an n x (m - 1) image with the seam removed.
    """
    #img[0][seam[0]:num-cols] = img[0]seam[0]+1]
    # Implement me!!!
    h,w,c = image.shape
    for i in range(0,len(seam)):
        image[i][seam[i]:w-2] = image[i][seam[i]+1:w-1]
    return image[:,:-1]

def gradient_magnitude(image):
    """
    Returns the L1 gradient magnitude of the given image.
    The result is an n x m numpy array of floating point values,
    where n is the number of rows in the image and m is the number of columns.
    """
    
    # 
    # First, convert the input image to a 32-bit floating point grayscale.
    # Be sure to scale it such that the intensity varies between 0 and 1.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get the min and max

    gray_img_float32 = np.float32(gray_image)/255.0
    # Next, compute the graient in the x direction using the sobel operator with a kernel size of 3.
    # Be sure to normalize the image so it's values are between -1 and 1
    horizontal_gradient = np.abs(cv2.Sobel(gray_img_float32, cv2.CV_64F, 1, 0, ksize=3)/4.0)

    # Compute the graient in the y direction using the same kernel size and normalization
    vertical_gradient = np.abs(cv2.Sobel(gray_img_float32, cv2.CV_64F, 0, 1, ksize=3)/4.0)
 
    # Finally, compute the l1 norm of the x and y gradients at each pixel value.
    # The l1 norm is the sum of their absolute values.
    # convert the final image from a double-precision floating point to single.

    abs_sum = horizontal_gradient + vertical_gradient 
    l1_norm = np.float32(abs_sum)
    # and return the result
    return(l1_norm)

def compute_seam_costs(energy):
    """
    Computes the cumulative minimum energy of every possible seam in the provided energy image.
    You can do this using the dynamic programming rule:
         M(i, j) = e(i, j) + min( M(i-1, j-1), M(i-1, j), M(i-1, j+1) 
    
    energy : an n x m single channel image defining the energy at each pixel.
    returns : an n x m image containing the cumulative minimum energy cost of all seams through each pixel.
    """

    # Create M, an n x m matrix with the first row equal to energy.
    dimensions = (len(energy),len(energy[0]))
    M = np.zeros(dimensions)
    for i in range(0,len(energy[0])):
        M[0][i] = energy[0][i]
    # Iterate over the rows, starting at row 1
    for i in range(1,len(energy)):
        #print(len(energy))
        # Iterate over the column 1 to m - 1
        for j in range(0,len(energy[0])):
            #print(len(energy[i]))
            if j == 0:
                M[i][j] = energy[i][j] + np.min([M[i-1][j], M[i-1][j+1]])
            elif j == (len(energy[0])-1):
                M[i][j] = energy[i][j] + np.min([M[i-1][j-1], M[i-1][j]])
            else:
                M[i][j] = energy[i][j] + np.min([M[i-1][j-1], M[i-1][j], M[i-1][j+1]])
        # Compute M(i, j) = e(i, j) + min( M(i-1, j-1), M(i-1, j), M(i-1, j+1)
        # be sure to handle the special cases where j = 0 and j = m -1

    # return the result!
    return(M)

def minimal_seam(M):
    """
    Find the seam with minimal energy cost given the provided seam cost
    matrix M. Returns the X-coordinates of the minimal-cost vertical seam in
    top-down order.
    
    M: the output from compute_seam_costs.
    
    return: a list of x-coordinates starting at row 0 of M containing the ones to remove.
    """

    # Compute the bottom-up path of pixel X-coordinates for the seam with
    # minimal cost.
    min_seam = []
    row_mins = []
    min_seam.insert(0,np.argmin(M[len(M)-1]))
    for i in range(len(M)-2, 0, -1):
        if min_seam[0] == 0:
            row_mins = M[i][min_seam[0]], M[i][min_seam[0] + 1]
            min_seam.insert(0,min_seam[0] + np.argmin(row_mins))
        elif min_seam[0] == len(M[0])-1:
            row_mins = M[i][min_seam[0]-1], M[i][min_seam[0]]
            min_seam.insert(0,(min_seam[0]-1) + np.argmin(row_mins))
        else:
            row_mins = M[i][min_seam[0]-1], M[i][min_seam[0]], M[i][min_seam[0]+1]
            min_seam.insert(0,(min_seam[0]-1) + np.argmin(row_mins))

    # Return the top-down seam X-coordinates and the total energy cost of
    # removing that seam.
    costs = M[len(M)-1][min_seam[len(min_seam)-1]]
    return min_seam, costs

def compute_ordering(image, target_size):
    """
    Compute the optimal order of horizontal and vertical seam removals to
    achieve the given target image size. Order should be returned as a list of
    0 or 1 values corresponding to horizontal and vertical seams
    respectively.
    """
    r = image.shape[0] - target_size[0] + 1
    c = image.shape[1] - target_size[1] + 1
    if r < 0 or c < 0:
        raise ValueError("Target size must be smaller than the input size.")
    return [0,1] * min(r-1, c-1) + [0] * max(r-c, 0) + [1] * max(c-r, 0)

def resize(image, target_size):
    output = image.copy()
    order = compute_ordering(output, target_size)

    for i, seam_type in enumerate(order):
        print "Removing seam {} / {} ".format(i, len(order))
        
        # check if order = 0, if so, transpose the image!
        if order[i] == 0:
            output = cv2.transpose(output)
        # compute the energy using gradient_magnitude
        energy = gradient_magnitude(output)
        # Compute M using compute_seam_costs
        image_energy = compute_seam_costs(energy)
        # get the minimal seam using 'minimal_seam'
        min_seam,costs = minimal_seam(image_energy)
        # remove it using 'remove_vertical_seam'
        output = remove_vertical_seam(output, min_seam)
        # check if order = 0, if so, transpose the image back!
        if order[i] == 0:
            output = cv2.transpose(output)
    # Sanity check.....
    print(output.shape)
    print(target_size)
    assert(output.shape[0] == target_size[0] and
           output.shape[1] == target_size[1])
           
    # return results...
    return output

def runDebug(image, target_size):
    # compute and save a gradient image
    energy = gradient_magnitude(image.copy()) / 2.0
    cv2.imwrite("debug-energy.png", energy * 255)
    print "energy min max {} {}".format(np.min(energy.flatten()), np.max(energy.flatten()))
    
    # compute and save the cost image
    cost = np.log(compute_seam_costs(energy) + 1.0)
    cv2.imwrite("debug-cost.png", cost * 255)
    
    # get the minimal seam using 'minimal_seam'
    seam, seamCost = minimal_seam(cost)
    print "The minimal seam has cost {}".format(seamCost)
    print "The first seam is {}".format(seam)
    
    # remove it using 'remove_vertical_seam'
    output = remove_vertical_seam(image.copy(), seam)
    cv2.imwrite("debug-first-seam-gone.png", output)

if __name__ == "__main__":
    debug = (len(sys.argv) > 5 and sys.argv[5] == "-debug")
    try:
        in_fn, w, h, out_fn = sys.argv[1:5]
        h, w = int(h), int(w)
    except:
        print("Usage: python hw1.py FILE TARGET_WIDTH TARGET_HEIGHT OUTPUT")
        exit(1)

    image = cv2.imread(in_fn)
    if (debug):
        runDebug(image, (h, w))
    else:
        resized = resize(image, (h,w))
        cv2.imwrite(out_fn, resized)
