import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage.filters as flt
import time
from vstru2mw import vstru2mw


#Aufgabenteil a

def apply_simple_max(img_in, img_out, s):
    '''
    Applies a simple maximum filter linewise

    Parameters
    ----------
    img_in : numpy.array
        input image.
    img_out : numpy.array
        output image with maximum values.
    s : int
        size of linear mask.

    Returns
    -------
    None.

    '''
    (h, w) = np.shape(img_in)
    s2 = np.floor(s / 2).astype(int)
    for y in range(0, h):
        mask = []
        mask.extend(np.zeros(s))
        mask[s2:s] = img_in[y][0:s2]
        img_out[y][0] = max(mask)

        for x in range(1, w):
            mask = mask[1:]
            if x + s2 + 1 >= w - 1:
                mask.append(0)
            else:
                mask.append(img_in[y][x + s2])

            img_out[y][x] = max(mask)

    return


def simple_max_filter(img, s):
    '''
    Applies a simple maximum filter in a rectangular neighborhood of size s by
    applying the linewise filter twice

    Parameters
    ----------
    img : numpy.array
        input image.
    s : int
        size of rectangular filter mask, only uneven values supported,
        even values for s are incremented by 1.

    Returns
    -------
    img_result : numpy.array
        maximum filtered image.

    '''
    if s % 2 == 0:
        s += 1

    (h, w) = np.shape(img)

    linewise_max_img = np.zeros([h, w])
    apply_simple_max(img, linewise_max_img, s)

    img_result = np.zeros([w, h])
    linewise_max_img = np.transpose(linewise_max_img)
    apply_simple_max(linewise_max_img, img_result, s)
    img_result = np.transpose(img_result)
    return img_result

img = cv2.imread('/Users/AEMMERICH/Desktop/Graubild.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#img_max = simple_max_filter(img, 7)

#fig = plt.figure()
#ax = fig.add_subplot(1, 2, 1)
#plt.imshow(img, cmap="gray")
#ax.set_title('Before')
#ax = fig.add_subplot(1, 2, 2)
#plt.imshow(img_max, cmap="gray")
#ax.set_title('After')
#fig.suptitle("Simple Max Filter", fontsize=16)
#plt.show()

#Aufgabenteil b

v = np.ndarray([0,0,3,2,1,1,0,0])
mw = vstru2mw(v)
print(mw)