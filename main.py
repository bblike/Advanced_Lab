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


#


def rgb2gray(rgb):
    gray = np.mean(rgb, -1)
    return gray


def calculation_red(q):
    temp = np.full((structure[0], structure[1]), 0)
    for i in range(0, red_structure[0]):
        for j in range(0, red_structure[1]):
            temp[i][j] = im[i][j][0]
            # red[i][j][1] = im[i][j][0]

        print("process{} done.".format(i))
    q.put(temp)


def calculation_red_param(param):
    temp = []
    for i in param:
        for j in range(0, red_structure[1]):
            red[i][j][0] = im[i][j][0]

        print("process {} done.".format(i))
    print("calculation {} down".format(param))


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
    print("results=", results)
    print("time_spent=", end_time - begin_time)
    return results


def print_image(x):
    cv.waitKey(5000)
    cv.imshow("new window", x)
    cv.waitKey(2000)
    cv.destroyAllWindows()
    cv.imwrite("raw-r.jpg", x)


path = r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thursdayw4"
files = os.listdir(path)
# path = "newtest.JPG"
file_length = len(files)
selector = [500]
excel_result = []
if __name__ == '__main__':
    for file in files:
        for i in selector:
            for j in range(0, 20):
                if file == "{}P{}.JPG".format(i, j):
                    im = cv.imread(r"thursdayw4\{}".format(file))
                    # global structure
                    structure = np.shape(im)
                    temp = rgb2gray(im)
                    print(structure)
                    result = []
                    # global red_structure
                    red_structure = np.array([structure[0], structure[1], 3])
                    # global red
                    red = np.zeros(red_structure)

                    # parallel()
                    # temp = threading_job()
                    print(temp)
                    print("structure of temp is ", np.shape(temp))
                    print("waiting...")
                    # temp1 = np.array(temp[0])
                    temp1 = np.array(temp)
                    new_im = Image.fromarray(temp1)
                    new_im.show()
                    #new_im.save("temp_{}P{}.jpg".format(i, j))
                    # goto the middle
                    xpos = int(structure[0] / 2)
                    ypos = int(structure[1] / 2)
                    r = 30
                    new_array = temp1[xpos - r:xpos + r, ypos - r:ypos + r]
                    print(new_array)
                    means = np.mean(new_array)
                    normaldis = 0
                    midx = int(len(new_array) / 2)
                    midy = int(len(new_array[0]) / 2)
                    for n in range(len(new_array)):
                        for m in range(len(new_array[n])):
                            arg = np.sqrt((midx - n) ** 2 + (midy - m) ** 2)
                            if arg == 0:
                                normaldis += new_array[n][m] * 2
                            elif arg <= 20:
                                normaldis += new_array[n][m] / arg
                    stds = np.std(new_array)
                    print("mean of {}{} = ".format(i, j), means)
                    print("standard error of {}{} =".format(i, j), stds)
                    print("normalised result", normaldis)
                    excel_result.append([i, j, means, stds, normaldis])

    print(excel_result)
    """writer = np.array(excel_result)
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet('thursday', cell_overwrite_ok=True)
    sheet.write(0, 1, "pressure")
    sheet.write(0, 2, "angle count")
    sheet.write(0, 3, "mean")
    sheet.write(0, 4, "standard error")
    sheet.write(0, 5, "normalised results")
    for i in range(len(writer)):
        for j in range(len(writer[0])):
            sheet.write(i + 1, j + 1, float("{}".format(writer[i][j])))

    book.save('test.xls')"""
