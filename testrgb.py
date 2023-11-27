import os
import cv2 as cv
import numpy as np


im = cv.imread("CIMG{}.JPG".format(4289))
cv.namedWindow("test", 0)
cv.imshow("test", im)
cv.waitKey(100000)
cv.destroyAllWindows()
