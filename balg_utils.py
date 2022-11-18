import timeit
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

#empty dictionary to store filepaths
filepaths = {}

def fill_filepaths():
    #list all filenames of images in data folder
    i = 0
    for filename in os.listdir('data'):
        #only add files with image extension
        if filename.endswith('.jpg' or '.png' or '.jpeg' or '.bmp'  or '.tif' or '.tiff'):
            filepaths[i] = os.path.join('data', filename)
            i += 1


def read_image(image_number, gray=True):
    img = cv2.imread(filename=filepaths.get(image_number))
    #convert to gray
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #if image is to big, half the size until either width or height is smaller than 1500
    while(img.shape[1]> 1500 or img.shape[0]> 1500):
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)   
        img = cv2.resize(img, dim)
    return img


def race_functions(func1, func2, num_runs, args):
    t_func1 = 0
    t_func2 = 0
    for i in range(num_runs):
        t_func1 += timeit.timeit(lambda: func1(*args), number=1)
        t_func2 += timeit.timeit(lambda: func2(*args), number=1)
    
    t_func1 /= num_runs
    t_func2 /= num_runs
    #round to 3 significant digits
    t_func1 = np.format_float_positional(t_func1, precision=4, unique=False, fractional=False, trim='k')
    t_func2 = np.format_float_positional(t_func2, precision=4, unique=False, fractional=False, trim='k')
    #print results
    print('Mean Execution Time for '+ func1.__name__+' : ' + t_func1 + 's, with ' + str(num_runs) + ' runs')
    print('Mean Execution Time for '+ func2.__name__+' : ' + t_func2 + 's, with ' + str(num_runs) + ' runs')
    return  t_func1, t_func2



def difference_image(img1, img2, name1='img1', name2='img2'):
    diff_img = np.abs(img1 - img2).astype(np.uint8)
    #print max gray value
    print('Max gray value in difference image: ' + str(np.max(diff_img)))
    #show both images and the difference
    plt.figure(figsize=(15, 10))
    plt.subplot(1,3,1)
    plt.imshow(img1, cmap='gray')
    plt.title(name1)
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap='gray')
    plt.title(name2)
    plt.subplot(1,3,3)
    plt.imshow(diff_img, cmap='gray')
    plt.title('Difference Image')
    plt.show()
    return diff_img


fill_filepaths()