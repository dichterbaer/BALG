import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
from vstru2mw import vstru2mw

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


#b

def max_filter_vnormal(img, s):
    if s % 2 == 0:
        raise ValueError('Mask size must be odd.')
    mask = vstru2mw(s) #mehrdimensionale mask

    img_v = simple_max_filter(img,mask)
    return img_v

img_vnormal = max_filter_vnormal(img, 7)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(img_vnormal, cmap="gray")
ax.set_title('v-normal')
plt.show()