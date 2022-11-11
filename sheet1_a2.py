import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def bresenham(aspectratio = 1):
    width = 50
    height = int(width * aspectratio)
    x = 0
    y = 0
    dx = width - x - 1
    dy = height - y - 1
    img = np.zeros((height,width,1), np.uint8)

    errorXdx = 0
    errorYdy = 0
    img[y,x] = 255
    operations = 0
    ops = []
    loops = []

    loopcounter = 0
    while x < width-1: 
        operations += 1 #vgl while
        errorXdx = errorXdx + dy
        errorYdy = errorYdy + dx
        operations += 2
        operations += 2 # vgl if
        if errorXdx + errorXdx > dx:
            y = y + 1
            errorXdx = errorXdx - dx
            operations += 2
        operations += 2 # vgl if
        if errorYdy + errorYdy > dy:
            x = x + 1
            errorYdy = errorYdy - dy
            operations += 2
        img[y,x] = 255
        operations += 1
        loopcounter += 1
        ops.append(operations)
        loops.append(loopcounter)
    return img, ops, loops


aspectratios = np.arange(0.05, 1.05, 0.05)
num_operations = []
images = []
for aspectratio in aspectratios:
    img, ops, loops = bresenham(aspectratio)
    num_operations.append(ops[-1])
    images.append(img)

# Plot the number of operations as a function of the aspect ratio
plt.plot(aspectratios, num_operations)
plt.xlabel('Aspect ratio')  
plt.ylabel('Number of operations')
plt.show()

#show all images in a grid
fig, axs = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):
        axs[i, j].imshow(images[i*5+j], cmap='gray')
        axs[i, j].axis('off')
plt.show()

# fig = plt.figure(figsize=(30, 15))
# ax = fig.add_subplot(1, 2, 1)
# ax.set_title('Bresenham\'s Line Algorithm')
# ax.imshow(img, cmap='gray', vmin = 0, vmax = 255)
# ax = fig.add_subplot(1, 2, 2)
# ax.set_title('Operations')
# ax.set_xlabel('Loops')
# ax.set_ylabel('Operations')
# ax.plot(loops,ops)
plt.show()
