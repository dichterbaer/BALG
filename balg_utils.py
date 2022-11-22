import timeit
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

#empty dictionary to store filepaths
filepaths = {}

def fill_filepaths():
    '''
    Fills the filepaths dictionary with the filepaths of the images in the folder 'data'
    '''
    #list all filenames of images in data folder
    i = 0
    for filename in os.listdir('data'):
        #only add files with image extension
        #if filename.endswith('.jpg' or '.png' or '.jpeg' or '.bmp'  or '.tif' or '.tiff'):
        if (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.bmp')  or filename.endswith('.tif') or filename.endswith('.tiff')):
            filepaths[i] = os.path.join('data', filename)
            i += 1



def read_image(image_identifier, gray=True):
    '''
    read image from file

    Parameters
    ----------
    image_identifier : string or int
        if string: path to image file
        if int: index of image in filepaths dictionary
    gray : bool, optional
        if True: read image as grayscale. The default is True.


    Returns
    -------
    img : numpy array
        image as numpy array
    '''
    if type(image_identifier) == str:
        #check if image exists
        if os.path.exists(image_identifier):
            img = cv2.imread(image_identifier)
        else:
            print('File not found')
            return
    elif type(image_identifier) == int:
        #check if index is in dictionary
        if image_identifier in filepaths:
            img = cv2.imread(filename=filepaths.get(image_identifier))
        else:
            print('Index not found')
            return
    #convert to gray
    if gray:
        #check if channels > 1
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #if image is to big, half the size until either width or height is smaller than 1500
    while(img.shape[1]> 1500 or img.shape[0]> 1500):
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)   
        img = cv2.resize(img, dim)
    return img


def race_functions(func1, func2, num_runs, args1, args2=None, verbose = False):
    '''
    Compares the runtime of two functions

    Parameters
    ----------
    func1 : function
        first function to compare
    func2 : function
        second function to compare
    num_runs : int
        number of times to run each function
    args1 : list
        list of arguments to pass to the functions
    args2 : list, optional
        list of arguments to pass to the function2 if different arguments are needed. The default is None.
    verbose : boolean, optional
        if True: show progress and result times. The default is False.

    Returns
    -------
    t_func1 : float
        average runtime of func1
    t_func2 : float
        average runtime of func2
    '''
    t_func1 = 0
    t_func2 = 0
    for i in range(num_runs):
        if args2 is None:
            t_func1 += timeit.timeit(lambda: func1(*args1), number=1)
            t_func2 += timeit.timeit(lambda: func2(*args1), number=1)
        else:
            t_func1 += timeit.timeit(lambda: func1(*args1), number=1)
            t_func2 += timeit.timeit(lambda: func2(*args2), number=1)
        #print progress
        if verbose: 
            print('Run ' + str(i+1) + ' of ' + str(num_runs) + ' finished')


    
    t_func1 /= num_runs
    t_func2 /= num_runs
    #round to 3 significant digits
    t_func1 = np.format_float_positional(t_func1, precision=4, unique=False, fractional=False, trim='k')
    t_func2 = np.format_float_positional(t_func2, precision=4, unique=False, fractional=False, trim='k')
    #print results
    if verbose:
        print('Mean Execution Time for '+ func1.__name__+' : ' + t_func1 + 's, with ' + str(num_runs) + ' runs')
        print('Mean Execution Time for '+ func2.__name__+' : ' + t_func2 + 's, with ' + str(num_runs) + ' runs')
    return  t_func1, t_func2



def difference_image(img1, img2, name1='img1', name2='img2'):
    '''
    Plots the difference between two images

    Parameters
    ----------
    img1 : numpy array
        first image
    img2 : numpy array
        second image
    name1 : string, optional
        name of first image. The default is 'img1'.
    name2 : string, optional
        name of second image. The default is 'img2'.

    Returns
    -------
    diff : numpy array
        difference image
    '''
    diff_img = np.abs(img1 - img2).astype(np.uint8)
    #print max gray value
    print('Max gray value in difference image: ' + str(np.max(diff_img)))
    #show both images and the difference
    plt.figure(figsize=(15, 10))
    plt.subplot(1,4,1)
    plt.imshow(img1, cmap='gray')
    plt.title(name1)
    plt.subplot(1,4,2)
    plt.imshow(img2, cmap='gray')
    plt.title(name2)
    plt.subplot(1,4,3)
    plt.imshow(diff_img, cmap='gray')
    plt.title('Difference Image')
    #show difference image scaled to 0-255
    plt.subplot(1,4,4)
    plt.imshow(diff_img, cmap='gray', vmin=0, vmax=255)
    plt.title('Difference Image, \nscaled to 0-255')
    plt.show()
    return diff_img

#fill filepaths dictionary
fill_filepaths()