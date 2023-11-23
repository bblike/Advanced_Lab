import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
path = r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\test (4).JPG"
im = cv.imread(path)

blue = np.array(im)
red = np.array(im)
green = np.array(im)

for i in range(len(im)):
    for j in range(len(im[0])):
        blue[i][j][2] = 0
        blue[i][j][1] = 0
        red[i][j][0] = 0
        red[i][j][1] = 0
        green[i][j][0] = 0
        green[i][j][2] = 0
        pass


print("waiting...")
cv.waitKey(1000)
cv.imshow("red", red)
cv.imshow("green", green)
cv.imshow("blue", blue)
cv.waitKey(1000000)
cv.destroyAllWindows()
