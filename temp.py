# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
def draw_average_lines(img, lines,color=[255,0,0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # use llines_mat to store left lines, use rlines_mat to store right lines
    llines_mat = np.empty(shape=(0,4),dtype=int)
    rlines_mat = np.empty(shape=(0,4),dtype=int)
    #i=0
    for line in lines:
        for x1,y1,x2,y2 in line:
            # debug use
            #print('x1=', x1,'y1=', y1, 'x2=', x2, 'y2=', y2, 'm=','right' if ((y2-y1)/(x2-x1))>0 else 'left')
            #lines_mat[i,:] = [x1,y1,x2,y2] 
            if ((y2-y1)/(x2-x1)) > 0:
                rlines_mat = np.append(rlines_mat, [[x1,y1,x2,y2]], axis=0)
            else:
                llines_mat = np.append(llines_mat, [[x1,y1,x2,y2]], axis=0)
            # draw all lines
            #cv2.line(img, (x1, y1), (x2, y2), [0,255,0], 2)
            #i = i+1
 
    # find average lines for left lines and right lines
    average_llines = np.average(llines_mat,axis=0).astype(int)
    average_rlines = np.average(rlines_mat,axis=0).astype(int)
    
    # the slope for average left line
    m_average_llines =  (average_llines[3]-average_llines[1])/(average_llines[2]-average_llines[0])
    # the slope for average right line
    m_average_rlines =  (average_rlines[3]-average_rlines[1])/(average_rlines[2]-average_rlines[0])
    
    # For left lines
    # find the y coordinate of uppermost point u_point from llines_mat
    y_coordinates = np.append(llines_mat[:,1],llines_mat[:,3],axis=0)
    u_point_y = np.min(y_coordinates)
    
    # based on u_point_y, extend average_llines to u_point_y
    u_point_x = (average_llines[2]+(u_point_y-average_llines[3])/m_average_llines).astype(int)
    

    b_point_y = image.shape[0]
    b_point_x = (average_llines[0]+(b_point_y-average_llines[1])/m_average_llines).astype(int)

    #draw the extented average left line on image
    blank_image = np.zeros_like(img)
    img_with_llines = cv2.line(blank_image, (b_point_x, b_point_y), (u_point_x, u_point_y), color, thickness) 
    #plt.imshow(weighted_img(img_with_llines, img, α=0.8, β=1., λ=0.))
    
    
    # For right lines
    # find the y coordinate of uppermost point u_point from rlines_mat
    y_coordinates = np.append(rlines_mat[:,1],rlines_mat[:,3],axis=0)
    u_point_y = np.min(y_coordinates)
    
    # based on u_point_y, extend average_llines to u_point_y
    u_point_x = (average_rlines[2]+(u_point_y-average_rlines[3])/m_average_rlines).astype(int)
    

    b_point_y = image.shape[0]
    b_point_x = (average_rlines[0]+(b_point_y-average_rlines[1])/m_average_rlines).astype(int)

    #draw the extented average right line on image
    img_with_lrlines = cv2.line(img_with_llines, (b_point_x, b_point_y), (u_point_x, u_point_y), color, thickness) 
    output_image = weighted_img(img_with_lrlines, img, α=0.8, β=1., λ=0.)
    plt.imshow(output_image)
    return output_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
imshape = image.shape
gray = grayscale(image)

plt.imshow(gray, cmap='gray') 
# Define a kernel size and apply Gaussian smoothing
kernel_size = 5

blur_gray = gaussian_blur(gray, kernel_size)

low_threshold = 50
high_threshold = 150
edges = canny(blur_gray, low_threshold, high_threshold)

img_height = image.shape[0]
img_width = image.shape[1]
"""
vertices = np.array([[[40, img_height],
                      [img_width/4, 3*img_height/5],
                      [3*img_width/4, 3*img_height/5],
                      [img_width - 40, img_height]]], dtype=np.int32 )
"""
vertices = np.array([[[40, img_height],
                      [img_width/2-50, 3*img_height/5],
                      [img_width/2+50, 3*img_height/5],
                      [img_width - 40, img_height]]], dtype=np.int32 )

# draw vertices
image = mpimg.imread('test_images/solidWhiteRight.jpg')
plt.plot(vertices[:,:,0], vertices[:,:,1],'ro')
plt.imshow(image)


"""
vertices = np.array([[vertice1, vertice2, vertice3, vertice4]], dtype=np.int32)
"""
masked_edges = region_of_interest(edges, vertices)
plt.imshow(masked_edges, cmap='gray')


# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 40     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 30 #minimum number of pixels making up a line
max_line_gap = 200    # maximum gap in pixels between connectable line segments

line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
plt.imshow(line_img, cmap='gray')

"""
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 40 #minimum number of pixels making up a line
max_line_gap = 10    # maximum gap in pixels between connectable line segments

# obtain result similar to "raw-lines-example.mp4"
line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
plt.imshow(line_img, cmap='gray')
"""

# use HoughLinesP() to compute lines
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

#print(lines.shape)

img = mpimg.imread('test_images/solidWhiteRight.jpg')
output_img = draw_average_lines(img, lines, [255, 0, 0], 10)
#store output_img
#mpimg.imsave("test_images/solidWhiteRight_output.jpg", output_img)

# Process images in folder test_images/
import os
test_images = os.listdir("test_images/")
for image_name in test_images:
    image = mpimg.imread("test_images/"+image_name)
    img_height = image.shape[0]
    img_width = image.shape[1]
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    masked_edges = region_of_interest(edges, vertices)
    plt.imshow(masked_edges, cmap='gray')
    
    #line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    #plt.imshow(line_img, cmap='gray')
    # use HoughLinesP() to compute lines
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    img = mpimg.imread("test_images/"+image_name)
    output_img = draw_average_lines(img, lines, [255, 0, 0], 10)
    #store output_img
    #mpimg.imsave("output/result_"+image_name, output_img)

#lines_mat = lines_mat.astype(int)

# process image in  challenge_img/
#challenge_images = os.listdir("challenge_img/")
image_name = '000001.jpg'

image = mpimg.imread("challenge_img/"+image_name)
plt.imshow(image)

img_height = image.shape[0]
img_width = image.shape[1]

vertices = np.array([[[40, img_height],
                      [img_width/2-50, 3*img_height/5],
                      [img_width/2+50, 3*img_height/5],
                      [img_width - 40, img_height]]], dtype=np.int32 )

"""# draw vertices
plt.plot(vertices[:,:,0], vertices[:,:,1],'ro')
plt.imshow(image)
"""

low_threshold = 80
high_threshold = 300

rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 25     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments

gray = grayscale(image)
blur_gray = gaussian_blur(gray, kernel_size)
edges = canny(blur_gray, low_threshold, high_threshold)
blur_gray = gaussian_blur(gray, kernel_size)
edges = canny(blur_gray, low_threshold, high_threshold)
masked_edges = region_of_interest(edges, vertices)
plt.imshow(masked_edges, cmap='gray')

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

img = mpimg.imread("test_images/"+image_name)
output_img = draw_average_lines(img, lines, [255, 0, 0], 10)
         
image = mpimg.imread('test_images/solidWhiteRight.jpg')
# draw right lines

for j in range(i):
    if ((lines_mat[j,3]-lines_mat[j,1])/(lines_mat[j,2]-lines_mat[j,0]))>0:
        # draw right lines
        cv2.line(image, (lines_mat[j,0], lines_mat[j,1]), (lines_mat[j,2], lines_mat[j,3]), [0,0,255], 5)
    else:
        # draw left lines
        cv2.line(image, (lines_mat[j,0], lines_mat[j,1]), (lines_mat[j,2], lines_mat[j,3]), [0,255,0], 5)

plt.imshow(image)




image = mpimg.imread('test_images/solidWhiteRight.jpg')
for j in range(i):
    cv2.line(image, (lines_mat[j,0], lines_mat[j,1]), (lines_mat[j,2], lines_mat[j,3]), [0,0,255], 5)
    
plt.imshow(image)

#draw_lines(image, lines, color=[255, 0, 0], thickness=2)

#weighted_image = weighted_img(lines,image, 0.5, 5.5, 0.)
#plt.imshow(weighted_image)
#plt.show()

import os
os.listdir("test_images/")

#Test on Images

#im'solidWhiteRight.jpg'