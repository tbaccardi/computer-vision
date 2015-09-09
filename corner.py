# Corner detector prototype algorithm

import cv2
import numpy as np
import scipy.ndimage.filters as filt
import scipy.signal as sig
import time

# create a display window
cv2.namedWindow('normal', cv2.WINDOW_NORMAL)
cv2.namedWindow('vertical', cv2.WINDOW_NORMAL)
cv2.namedWindow('horizontal', cv2.WINDOW_NORMAL)
cv2.namedWindow('both', cv2.WINDOW_NORMAL)

# import picture from working directory
pix = cv2.imread("arch.jpg")
pix = cv2.resize(pix, (540, 450))

# convert picture to black and white format
bw_pix = cv2.cvtColor(pix, cv2.COLOR_RGB2GRAY)

# create Sobel masks for edge detection
# Canny edge detection is more accurate; however this is a simple demo
sobel_v = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
sobel_h = np.transpose(sobel_v) # just being lazy with the mask creation

# implement a gaussian blurring mask on the image
# this can come before or after the edge detection algorithm
gssn_pix = cv2.GaussianBlur(bw_pix, (5,5), 1)

# apply sobel masks to the filtered image
sv = sig.convolve(gssn_pix, sobel_v)
sh = sig.convolve(gssn_pix, sobel_h)

print np.min(sv)
# normalize the image; had to fall back to opencv funtion to save time
sv = cv2.convertScaleAbs(sv)
sh = cv2.convertScaleAbs(sh)

thresh = 230

vlow_val_index = sv << thresh
hlow_val_index = sh << thresh
vhigh_val_index = sv >> thresh
hhigh_val_index = sh >> thresh

sv[vlow_val_index] = 0
sh[hlow_val_index] = 0
sv[vhigh_val_index] = 255
sv[hhigh_val_index] = 255

# this essentially merges the two derivative images together! try it wihout this
# funtion and see what happens
total = cv2.addWeighted(sv, 0.5, sh, 0.5, 0)

svh = cv2.convertScaleAbs(total)

# this is where the harris corner detector comes in. now we are going to
# initialize a matrix to store all of the R-score values
E = np.zeros((len(sv),len(sv[0])))

# this loop will iterate through the values of the matrix and find R-scores and 
# insert them into the E matrix

k = 0.05 # empirical constant for the R-score calculation

for i in range(len(svh)):
	for j in range(len(svh[0])):
		ixx = np.uint8(sv[i,j])*np.uint8(sv[i,j])
		iyy = np.uint8(sh[i,j])*np.uint8(sh[i,j])
		ixy = np.uint8(sv[i,j])*np.uint8(sh[i,j])
		# the M matrix is a 2x2 matrix used to hold the pixel values
		# to find eigenvalues for the R-score
		M_mat = np.array([[ixx, ixy],[ixy, iyy]]) # in this case, the window function is just 1
		lambda_ = np.linalg.eigvals(M_mat)
		
		det = lambda_[0]*lambda_[1]
		trace = lambda_[0] + lambda_[1]
		
		R_score = det - k*trace*trace
	
		# in order to be computationally efficient, need to threshold R
		# here. 
		R_thresh = 40000
		if(R_score > R_thresh):
			R_score = 255
		else:
			R_score = 0
		E[i,j] = R_score

# Draw corners on original image
for i in range(len(E)):
	for j in range(len(E[0])):
		if(E[i,j]  == 255):
			cv2.circle(pix, (j,i), 25, (255,0,0))

# display all images
print len(sv), len(sv[0])
print len(pix), len(pix[0])
cv2.imshow('normal', pix)
cv2.imshow('vertical', sv)
cv2.imshow('horizontal',sh)
cv2.imshow('both', E)
cv2.waitKey()
cv2.destroyAllWindows()

