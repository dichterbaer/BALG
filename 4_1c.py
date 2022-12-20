from balg_utils import read_image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage as sk
from skimage.color import label2rgb
from scipy.ndimage.filters import maximum_filter
from skimage.morphology import skeletonize




def watershed_lokal_maxima_no_bg(image, verbose = True):
    
    #morphological smoothing
    kernelsize_morph_smooth = 5
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize_morph_smooth, kernelsize_morph_smooth))
    image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, disk)
    image_smoothed = cv2.morphologyEx(image_opened, cv2.MORPH_CLOSE, disk)

    kernelsize_maxfilter = 101
    filtered_image = maximum_filter(image_smoothed, size=kernelsize_maxfilter)

    # Find the local maxima by comparing the filtered image to the original image
    local_maxima = (filtered_image == image_smoothed)
    local_maxima_temp = local_maxima.copy()
    #dilate to connect neighboring blobs
    kernelsize_dilate = 21
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize_dilate, kernelsize_dilate))
    local_maxima = cv2.dilate(local_maxima.astype(np.uint8), disk)
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('b4 dilation')
        plt.imshow(local_maxima_temp, cmap=plt.cm.gray)
        plt.subplot(1,2,2)
        plt.title('after dialtion')
        plt.imshow(local_maxima, cmap=plt.cm.gray)
        plt.show()
    marker = local_maxima.astype(int)
    marker_labeled = cv2.connectedComponents(marker.astype(np.uint8))[1]
    
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('image_smoothed')
        plt.imshow(image_smoothed, cmap=plt.cm.gray)
        plt.subplot(1,2,2)
        plt.title('local_maxima')
        plt.imshow(marker_labeled)
        plt.show()

    kernelsize_sobel = 3
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernelsize_sobel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernelsize_sobel)

    # Combine the x and y gradients to get the gradient map
    gradient_map = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Normalize the gradient map to the range [0, 255]
    normalized_gradient_map = (gradient_map / gradient_map.max()) * 255
    kernelsize_gaussian_blur = 5
    normalized_gradient_map = cv2.GaussianBlur(normalized_gradient_map, (kernelsize_gaussian_blur,kernelsize_gaussian_blur), 0)
    
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('image')
        plt.imshow(image, cmap=plt.cm.gray)
        plt.subplot(1,2,2)
        plt.title('normalized_gradient_map')
        plt.imshow(normalized_gradient_map, cmap=plt.cm.gray)
        plt.show()

    watershed_no_backgroundmarkers_localmaxima = sk.segmentation.watershed(normalized_gradient_map, marker_labeled)
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('image')
        plt.imshow(image, cmap=plt.cm.gray)
        plt.subplot(1,2,2)
        plt.title('watershed wo backgroundmarkers and localmaxima')
        plt.imshow(watershed_no_backgroundmarkers_localmaxima, cmap=plt.cm.gray)
        plt.show()
    return watershed_no_backgroundmarkers_localmaxima

def watershed_threshold_no_bg(image, verbose = True):
    return image


def watershed_lokal_maxima_bg_marker(image, verbose = True):
#morphological smoothing
    kernelsize_morph_smooth = 5
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize_morph_smooth, kernelsize_morph_smooth))
    image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, disk)
    image_smoothed = cv2.morphologyEx(image_opened, cv2.MORPH_CLOSE, disk)

    kernelsize_maxfilter = 101
    filtered_image = maximum_filter(image_smoothed, size=kernelsize_maxfilter)

    # Find the local maxima by comparing the filtered image to the original image
    local_maxima = (filtered_image == image_smoothed)
    local_maxima_temp = local_maxima.copy()
    #dilate to connect neighboring blobs
    kernelsize_dilate = 21
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize_dilate, kernelsize_dilate))
    local_maxima = cv2.dilate(local_maxima.astype(np.uint8), disk)
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('b4 dilation')
        plt.imshow(local_maxima_temp, cmap=plt.cm.gray)
        plt.subplot(1,2,2)
        plt.title('after dialtion')
        plt.imshow(local_maxima, cmap=plt.cm.gray)
        plt.show()
    marker = local_maxima.astype(int)
    marker_labeled = cv2.connectedComponents(marker.astype(np.uint8))[1]
    
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('image_smoothed')
        plt.imshow(image_smoothed, cmap=plt.cm.gray)
        plt.subplot(1,2,2)
        plt.title('local_maxima')
        plt.imshow(marker_labeled)
        plt.show()

    kernelsize_sobel = 3
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernelsize_sobel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernelsize_sobel)

    # Combine the x and y gradients to get the gradient map
    gradient_map = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Normalize the gradient map to the range [0, 255]
    normalized_gradient_map = (gradient_map / gradient_map.max()) * 255
    kernelsize_gaussian_blur = 5
    normalized_gradient_map = cv2.GaussianBlur(normalized_gradient_map, (kernelsize_gaussian_blur,kernelsize_gaussian_blur), 0)
    
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('image')
        plt.imshow(image, cmap=plt.cm.gray)
        plt.subplot(1,2,2)
        plt.title('normalized_gradient_map')
        plt.imshow(normalized_gradient_map, cmap=plt.cm.gray)
        plt.show()
    #watershed with backgroundmarkers
    marker = marker.astype(np.uint8)
    markers_labeled = cv2.connectedComponents(marker)[1]
    marker_labeled = markers_labeled + 1
    markers_inv = cv2.bitwise_not(marker)
    markers_inv[markers_inv==255]=1
    markers_inv[markers_inv==254]=0
    bg_marker = skeletonize(markers_inv)

    marker_labeled = marker_labeled + bg_marker
    marker_labeled[marker_labeled>1]=255
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('marker')
        plt.imshow(marker)
        plt.subplot(1,2,2)
        plt.title('bg_marker + marker')
        plt.imshow(marker_labeled, cmap=plt.cm.gray)
        plt.show()

    #watershed
    watershed_backgroundmarkers_localmaxima = sk.segmentation.watershed(normalized_gradient_map, -marker_labeled)
    if(verbose):
        plt.figure(figsize=(15, 10))
        plt.subplot(1,2,1)
        plt.title('image')
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.title('watershed')
        plt.imshow(watershed_backgroundmarkers_localmaxima)
        plt.show()
    return watershed_backgroundmarkers_localmaxima



def watershed_threshold_bg_marker(image, verbose = True):
    return image


image = read_image('data/pears.png')

aufgabe = '1d'
if(aufgabe=='1c'):
    #gradient
    #image = cv2.GaussianBlur(image, (21,21), 0)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the x and y gradients to get the gradient map
    gradient_map = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Normalize the gradient map to the range [0, 255]
    normalized_gradient_map = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #gradient_smoothed = cv2.GaussianBlur(gradient, (19,19), 0)
    gradient_smoothed = normalized_gradient_map.astype(np.int32)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #watershed = cv2.watershed(gradient_smoothed)
    watershed = sk.segmentation.watershed(gradient_smoothed)
    rgb_image = label2rgb(watershed, bg_label=0, bg_color=(0, 0, 0))

    #show gradient and watershed
    plt.figure(figsize=(15, 10))
    plt.subplot(1,2,1)
    plt.title('Smoothed Gradient')
    plt.imshow(gradient_smoothed, cmap=plt.cm.gray)
    plt.subplot(1,2,2)
    plt.title('Oversegmentation with watershed')
    plt.imshow(rgb_image)
    plt.show()


if(aufgabe=='1d'):
    image = read_image('data/pears.png')
    # watershed_lclMax_no_bg = watershed_lokal_maxima_no_bg(image, verbose = True)
    # watershed_tr_no_bg = pears_threshold_no_bg(image, verbose = False)
    watershed_lclMax_bg_marker = watershed_lokal_maxima_bg_marker(image, verbose = True)
    # watershed_tr_bg_marker = pears_threshold_bg_marker(image, verbose = False)

    #show image and the 4 results
    plt.figure(figsize=(15, 10))
    plt.subplot(2,3,1)
    plt.title('image')
    plt.imshow(image, cmap=plt.cm.gray)
    plt.subplot(2,3,2)
    plt.title('watershed wo backgroundmarkers and localmaxima')
    plt.imshow(watershed_lclMax_no_bg, cmap=plt.cm.gray)
    plt.subplot(2,3,3)
    plt.title('watershed wo backgroundmarkers and threshold')
    plt.imshow(watershed_tr_no_bg, cmap=plt.cm.gray)
    plt.subplot(2,3,4)
    plt.title('watershed with backgroundmarkers and localmaxima')
    plt.imshow(watershed_lclMax_bg_marker, cmap=plt.cm.gray)
    plt.subplot(2,3,5)
    plt.title('watershed with backgroundmarkers and threshold')
    plt.imshow(watershed_tr_bg_marker, cmap=plt.cm.gray)
    plt.show()




   

    

