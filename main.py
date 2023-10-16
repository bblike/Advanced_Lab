# Numpy, Matplotlib and Pillow was used. Remember to import.
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
import multiprocessing as mp
import datetime
import math
import rgb2gray
import interactive as inte
from PIL import Image


def train_on_parameter(name, param):
    result = 0
    for num in param:
        result += math.sqrt(num * math.tanh(num) / math.log2(num) / math.log10(num))
    return {name: result}


def calculation():
    for i in range(0, raw_structure[0] - 1):
        for j in range(0, raw_structure[1] - 1):
            for k in range(0, 2):
                raw[i][j][k] = np.std([im[i][j][k], im[i][j + 1][k], im[i + 1][j][k], im[i + 1][j + 1][k]])

                print("process{}, {}, {} done.".format(i, j, k))


def calculation_r(param):
    print("holding calculations of ",len(param))
    for i in range(0, raw_structure[0] - 1):
        for j in param:
            for k in [0]:
                raw_r[i][j] = np.std([im[i][j][k], im[i][j + 1][k], im[i + 1][j][k], im[i + 1][j + 1][k]])
                print("process{}, {}, {} done.".format(i, j, k))


def calculation_empty(param):
    print("holding calculations of ",len(param))
    result = []
    for i in range(0, raw_structure[0] - 1):
        for j in param:
            for k in [0]:
                result = result + np.std([im[i][j][k], im[i][j + 1][k], im[i + 1][j][k], im[i + 1][j + 1][k]])
                print("process{}, {}, {} done.".format(i, j, k))





def calculation_matlab(image):
    print(np.shape(image))
    slope_x, slope_y = np.gradient(image)
    _field = np.sqrt(slope_x**2 + slope_y**2)
    return _field


#introducing the picture
path = "CIMG9548-1.JPG"
im = cv.imread(path)
print(np.shape(im))
structure = np.shape(im)

#cv.imshow("display window", im)
#cv.waitKey(1000)
#cv.destroyAllWindows()

raw = np.zeros(structure)
raw_r = np.zeros(structure)
raw_structure = np.array([structure[0] - 1, structure[1] - 1, 3])
div = int(structure[1] / 8)
raw_empty = []


inte.create_window(path)


"""
if __name__ == '__main__':

    start_t = datetime.datetime.now()

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)
    param_dict = {'task1': list(range(0, div)),
                  'task2': list(range(div + 1, div * 2)),
                  'task3': list(range(div * 2 + 1, div * 3)),
                  'task4': list(range(div * 3, div * 4)),
                  'task5': list(range(div * 4 + 1, div * 5)),
                  'task6': list(range(div * 5 + 1, div * 6)),
                  'task7': list(range(div * 6 + 1, div * 7)),
                  'task8': list(range(div * 7 + 1, div * 8))}
    results = [pool.apply_async(calculation_r, args=param) for param in param_dict.items()]
    results = [p.get() for p in results]

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")

cv.imshow("new window", raw_r)
cv.waitKey(1000000)
cv.destroyAllWindows()
cv.imwrite("raw-r.jpg", raw_r)
cv.imwrite("raw-r.txt", raw_r)


results = np.zeros([3, np.shape(im)[0], np.shape(im)[1]])
for k in range(0,3):

    results[k] = calculation_matlab(im[:,:,k])
    print(results[k])
    if k == 1:
        cv.imshow("original{}".format(k), im[:,:,k])


        cv.imshow("test{}".format(k), results[k])

        cv.imwrite("test{}.jpg".format(k), results[k])

while True:
    cv.waitKey(1000000)
cv.destroyAllWindows()
"""
