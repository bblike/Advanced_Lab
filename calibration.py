# Numpy, Matplotlib and Pillow was used. Remember to import.
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp
import datetime
import math
import time

from PIL import Image
import threading as td
from queue import Queue
import os
import xlwt
import calculation2
import calculation3 as c3

#


def rgb2gray(rgb):
    gray = np.mean(rgb, -1)
    return gray


def calculation_red(q):
    temp = np.full((structure[0], structure[1]), 0)
    for i in range(0, red_structure[0]):
        for j in range(0, red_structure[1]):
            temp[i][j] = im[i][j][2]
            # red[i][j][1] = im[i][j][0]

        print("process{} done.".format(i))
    q.put(temp)


def calculation_red_param(param):
    temp = []
    for i in param:
        for j in range(0, red_structure[1]):
            red[i][j][0] = im[i][j][0]

        print("process {} done.".format(i))
    #print("calculation {} down".format(param))


def parallel():
    begin_time = time.time()
    n = structure[0]
    procs = []
    n_cpu = int(mp.cpu_count() / 2)
    chunk_size = int(n / n_cpu)
    x = 0
    for i in range(0, n_cpu):
        min_i = chunk_size * i

        if i < n_cpu - 1:
            max_i = chunk_size * (i + 1)
        else:
            max_i = n
        digits = []
        for digit in range(min_i, max_i):
            digits.append(digit)
        print(digits)
        print(len(digits))
        procs.append(mp.Process(target=calculation_red_param, args=([digits])))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    end_time = time.time()
    print("time spend", end_time - begin_time)


def threading_job():
    q = Queue()
    ts = []
    begin_time = time.time()
    for i in range(0, 4):
        t = td.Thread(target=calculation_red, args=(q,))
        t.start()
        ts.append(t)

    for i in ts:
        i.join()
    results = []

    results.append(q.get())
    end_time = time.time()
    print("all done")
    #print("results=", results)
    print("time_spent=", end_time - begin_time)
    return results


def print_image(x):
    cv.waitKey(5000)
    cv.imshow("new window", x)
    cv.waitKey(2000)
    cv.destroyAllWindows()
    cv.imwrite("raw-r.jpg", x)


path = r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\calibration"
files = os.listdir(path)
# path = "newtest.JPG"
file_length = len(files)
xs = list(np.arange(360, 317.5, -2.5))
for i in range(len(xs)):
    xs[i] = round(xs[i], 1)

print("xs=", xs)

excel_result = []


def count_centre_uniformly(new_array, r):
    temp_array = []
    midx = int(len(new_array) / 2)
    midy = int(len(new_array[0]) / 2)
    for n in range(len(new_array)):
        for m in range(len(new_array[n])):
            arg = np.sqrt((midx - n) ** 2 + (midy - m) ** 2)
            if arg == 0:
                temp_array.append(new_array[n][m])
            elif arg <= r:
                temp_array.append(new_array[n][m])

    return np.array(temp_array)




if __name__ == '__main__':
    for file in files:
        for i in xs:
            if file == "{}.JPG".format(i):
                # read the image
                im = cv.imread(r"calibration\{}".format(file))
                # global structure
                structure = np.shape(im)
                # temp = rgb2gray(im)  # grey
                # print(structure)
                result = []
                red_structure = np.array([structure[0], structure[1], 3])
                red = np.zeros(red_structure)

                # parallel()
                temp = threading_job() #red
                # print(temp)
                # print("structure of temp is ", np.shape(temp))
                # print("waiting...")
                temp1 = np.array(temp[0]) #red
                #temp1 = np.array(temp)  # gray
                new_im = Image.fromarray(temp1)
                # new_im.show()
                # new_im.save("temp_{}P{}.jpg".format(i, j))
                # goto the middle
                xpos = int(structure[0] / 2)
                ypos = int(structure[1] / 2)
                r = 100
                new_array = temp1[xpos - r:xpos + r, ypos - r:ypos + r]
                print(new_array)
                print(np.shape(new_array))
                #print(new_array)
                means = np.mean(new_array)

                stds = np.std(new_array)/r
                #print("mean of {}{} = ".format(i, j), means)
                #print("standard error of {}{} =".format(i, j), stds)

                excel_result.append([i, means, stds])


    # save as excel files
    print(excel_result)
    xval = []
    yval = []
    yerr = []
    #print("temp=", temp)
    for element in excel_result:
            xval.append(element[0])
            yval.append(element[1])
            yerr.append(element[2])
    C, C_error = calculation2.calculation(np.array(xval), np.array(yval), np.array(yerr))

    print("the minimin angle is {}+-{}.".format(C, C_error))
