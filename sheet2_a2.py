import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from skimage.morphology import disk
from skimage.filters.rank import maximum
from vstru2mw import vstru2mw
from balg_utils import read_image, difference_image, race_functions2, race_functions

img = cv2.imread('data/baum_grau.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#a

def apply_simple_max(img_in, img_out, s):
    (h, w) = np.shape(img_in) #Höhe, Breite
    s2 = np.floor(s / 2).astype(int) #Setze s2 auf die Hälfte von s (abgerundet)
    for y in range(0, h):
        mask = []
        mask.extend(np.zeros(s))
        mask[s2:s] = img_in[y][0:s2]
        img_out[y][0] = max(mask) #Setze Anfang Bild aus Maximum der Maske

        for x in range(1, w): #Gehe durch Spalten
            mask = mask[1:]
            if x + s2 + 1 >= w - 1: #Falls Maske aus dem Bild raus
                mask.append(0)
            else:
                mask.append(img_in[y][x + s2])

            img_out[y][x] = max(mask)

    return


def simple_max_filter(img, s):


    if s % 2 == 0:
        raise ValueError('Mask size must be odd.')

    (h, w) = np.shape(img) #Höhe, Breite von input image

    #maximum filter erst zeilenweise, dann spaltenweise
    linewise_max_img = np.zeros([h, w]) #Erstelle initial max image mit Nullern
    apply_simple_max(img, linewise_max_img, s) #img auf linewise_max_img

    img_result = np.zeros([w, h])
    linewise_max_img = np.transpose(linewise_max_img)
    apply_simple_max(linewise_max_img, img_result, s) #linewise_max_img auf img_result
    img_result = np.transpose(img_result)
    return img_result
    


def max_filter_v_normal(img, mask_in):
    # # check if the mask size is odd
    # if mask_in.shape[0] % 2 == 0:
    #     raise ValueError('Mask size must be odd')
    mask = vstru2mw(mask_in)
    # create a new image with the same size as the input image
    max_img = np.zeros(img.shape)
    # filter_height = mask.shape[0]
    # filter_width = mask[]]

    #create empty histogram with 256 bins
    hist = np.zeros(256)
    #iterate over the image lines and ignore the border pixels
    for j in range(int(mask[0][0]), img.shape[0] - int(mask[0][0])):
        #initialize the histogram for the line
        init_histogram_v_normal(img, hist, int(mask[0][0]), j, mask)
        #iterate over the line and ignore the border pixels
        for i in range(mask.shape[0]//2, img.shape[1] - mask.shape[0]//2):
            #calculate the max of the histogram
            #return indice of max value in histogram
            max_img[j, i] = max(np.argwhere(hist>0))[0]
            #update the histogram for the next iteration
            update_histogram_v_normal(img, hist, i, j, mask)
    return max_img


def init_histogram_v_normal(img, hist, x, y, filter):
    #fill histogramm with 0
    hist.fill(0)
    #iterate over the mask and fill histogram
    k=0
    for j in range(-(filter.shape[0]//2), filter.shape[0]//2 + 1):
        #for i in range(-int(filter[k][0]-filter[k][2]), int(filter[k][0]-filter[k][2])+1):
        #nicht pixelweise durchgehen sondern mit np(:)
        #calc start and end index
        x_start = int(x-(filter[k][1]//2))
        x_end = int(x+(filter[k][1]//2))+1
        hist[img[y+j, x_start:x_end]] += 1
        k+=1
    return


def update_histogram_v_normal(img, hist, x, y, filter):
    #update histogram
    k=0
    for j in range(int(-filter[k][2]), int(filter[k][2]) + 1):
        hist[img[ y + j, x - (int(filter[k][1]//2))]] -= 1
        if(x + (int(filter[k][1]//2))+1 < img.shape[1]):
            hist[img[y + j, x + (int(filter[k][1]//2))+1]] += 1
        k+=1
    return


def max_vnormal_pad(img, mask):
    w2 = mask.shape[0] // 2
    im_padded = np.pad(img, ((w2, w2), (w2, w2)), 'constant', constant_values=0)
    im_out = max_filter_v_normal(im_padded, mask)
    return im_out[w2:-w2, w2:-w2]



'''
img_max = simple_max_filter(img, 21)
img_cv = scipy.ndimage.maximum_filter(img, 21)
#difference
diff = abs(img_max - img_cv)
print(np.max(np.array(diff)))

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
ax.set_title('Before')
ax = fig.add_subplot(1, 2, 2)
plt.imshow(img_max, cmap="gray")
ax.set_title('After')
fig.suptitle("Simple Max Filter", fontsize=16)

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
plt.imshow(img_max, cmap="gray")
ax.set_title('Simple Maximum Filter')
ax = fig.add_subplot(1, 3, 2)
plt.imshow(img_cv, cmap="gray")
ax.set_title('Maximum Filter von scipy')
ax = fig.add_subplot(1, 3, 3)
plt.imshow(diff, cmap="gray")
ax.set_title('Difference')
fig.suptitle("Comparison Filters & Difference", fontsize=16)
plt.show()
'''

testarray = [[10,5,6,20,4,10,8],
            [15,1,7,2,9,11,7],
            [18,4,3,6,8,10,12],
            [1,6,7,8,9,20,21],
            [10,15,18,2,1,3,4],
            [20,14,15,9,2,1,10],
            [18,4,3,6,8,10,12]]

img = np.array(testarray).astype(np.uint8)


img = read_image(1)
radius = 2
mask = disk(radius)
img_max = max_filter_v_normal(img, mask)
img_max_padded = max_vnormal_pad(img, mask)
img_max_sk = maximum(img, mask)
img_max_scipy = maximum_filter(img, footprint = mask)
diff = abs(img_max_padded - img_max_scipy)
#store all indices where difference is not 0
diff_indices = np.argwhere(diff != 0)
#difference
# difference_image(img_max, img_sk)
difference_image(img_max_scipy, img_max_padded)


# for radius in range(2, 21):
#     mask = disk(radius)
#     # times_ownImplementation, times_scipy = race_functions2((max_vnormal_pad, maximum_filter), ((img, mask), (img, mask)), 20)
#     times_ownImplementation, times_scipy = race_functions(max_vnormal_pad, maximum_filter, (img, mask), (img, mask), 20)
#     #difference

# # plot times over radius
# plt.plot(range(2, 21), times_ownImplementation, label="own implementation")
# plt.plot(range(2, 21), times_scipy, label="scipy implementation")
# plt.xlabel("radius")    
# plt.ylabel("time in ms")
# plt.legend()
# plt.show()

