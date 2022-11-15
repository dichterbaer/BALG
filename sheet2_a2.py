import numpy as np
import cv2
from vstru2mw import vstru2mw


#Aufgabenteil a

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

    if s % 2 == 0: #size Filtermaske (nur ungerade - bei gerade: addiere 1)
        s += 1

    (h, w) = np.shape(img) #Höhe, Breite von input image

    linewise_max_img = np.zeros([h, w]) #Erstelle initial max image mit Nullern
    apply_simple_max(img, linewise_max_img, s) #img auf linewise_max_img

    img_result = np.zeros([w, h])
    linewise_max_img = np.transpose(linewise_max_img)
    apply_simple_max(linewise_max_img, img_result, s) #linewise_max_img auf img_result
    img_result = np.transpose(img_result)
    return img_result

img = cv2.imread('/Users/AEMMERICH/Desktop/Graubild.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#img_max = simple_max_filter(img, 7)
"""fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
ax.set_title('Before')
ax = fig.add_subplot(1, 2, 2)
plt.imshow(img_max, cmap="gray")
ax.set_title('After')
fig.suptitle("Simple Max Filter", fontsize=16)
plt.show()"""


#Aufgabenteil b

v = np.ndarray([0,0,3,2,1,1,0,0])
mw = vstru2mw(v)
print(mw)