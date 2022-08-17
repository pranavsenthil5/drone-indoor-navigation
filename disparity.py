import cv2

import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('images/left1.jpeg', 0)
imgR = cv2.imread('images/right1.jpeg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')


plt.savefig('disparity.png', format='png')
plt.show()
plt.close()

# import numpy as np
# import cv2

# # Load the left and right images in gray scale
# # imgLeft = cv2.imread('../data/tsukuba_l.png', 0)
# # imgRight = cv2.imread('../data/tsukuba_r.png', 0)
# imgLeft = cv2.imread('images/left1.jpeg', 0)
# imgRight = cv2.imread('images/right1.jpeg', 0)

# print(type(imgLeft))

# # Initialize the stereo block matching object
# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=13)
# # stereo = cv2.StereoBM_create(numDisparities=0, blockSize=21)
# # Compute the disparity image
# disparity = stereo.compute(imgLeft, imgRight)

# # Normalize the image for representation
# min = disparity.min()
# max = disparity.max()
# disparity = np.uint8(255 * (disparity - min) / (max - min))

# # Display the result
# cv2.imshow('disparity', np.hstack((imgLeft, imgRight, disparity)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
