import cv2

import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('images/left3.jpeg', 0)
imgR = cv2.imread('images/right3.jpeg', 0)
# for i in range(0, imgL.shape[0]):

for i in range(0, 15):
    # print(i)
    stereo = cv2.StereoBM_create(numDisparities=16*i, blockSize=21)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity, 'autumn')

    plt.savefig('output/disparity_' + str(i) + '.png', format='png')
    plt.show()
    plt.close()
