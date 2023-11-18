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
import calibration as cali
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


path = r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thursdayw4"
files = os.listdir(path)
# path = "newtest.JPG"
file_length = len(files)
selector = [100, 200, 300, 4000, 500, 600]
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


def count_around(array, pos):
    return result


if __name__ == '__main__':
    for file in files:
        for i in selector:
            for j in range(0, 20):
                if file == "{}P{}.JPG".format(i, j):
                    # read the image
                    im = cv.imread(r"thursdayw4\{}".format(file))
                    # global structure
                    structure = np.shape(im)
                    temp = rgb2gray(im)  # grey
                    # print(structure)
                    result = []
                    # global red_structure
                    red_structure = np.array([structure[0], structure[1], 3])
                    # global red
                    red = np.zeros(red_structure)

                    #
                    # parallel()
                    #temp = threading_job() #red
                    # print(temp)
                    # print("structure of temp is ", np.shape(temp))
                    # print("waiting...")
                    #temp1 = np.array(temp[0]) #red
                    temp1 = np.array(temp)  # gray
                    new_im = Image.fromarray(temp1)
                    # new_im.show()
                    # new_im.save("temp_{}P{}.jpg".format(i, j))
                    # goto the middle
                    xpos = int(structure[0] / 2)
                    ypos = int(structure[1] / 2)
                    r = 12
                    new_array = temp1[xpos - r:xpos + r, ypos - r:ypos + r]
                    #print(new_array)
                    r_circle = 10   # radius of the circle selected
                    new_generated = count_centre_uniformly(new_array, r_circle)
                    means = np.mean(new_generated)
                    # normaldis = count_centre(new_array)
                    stds = np.std(new_generated)/np.sqrt(len(new_generated))
                    #print("mean of {}{} = ".format(i, j), means)
                    #print("standard error of {}{} =".format(i, j), stds)
                    # print("normalised result", normaldis)
                    excel_result.append([i, j, means, stds])
                    # excel_result.append([i, j, means, stds, normaldis])

    # save as excel files
    #print(excel_result)



    new = []
    group100 = []
    group200 = []
    group300 = []
    group400 = []
    group500 = []
    group600 = []

    for element in excel_result:
        match element[0]:
            case 100:
                group100.append(element)
            case 200:
                group200.append(element)
            case 300:
                group300.append(element)
            case 4000:
                group400.append(element)
            case 500:
                group500.append(element)
            case 600:
                group600.append(element)

    #groups = [[group600]]
    groups = [[group100], [group200], [group300], [group400], [group500], [group600]]
    for temp in groups:
        #print("temp=", temp)
        temp1 = temp[0]
        xval = []
        yval = []
        yerr = []
        for element in temp1:
            xval.append(element[1])
            yval.append(element[2])
            yerr.append(element[3])

        fix = 0
        match temp1[0][0]:
            case 100:
                fix = -20+360
            case 200:
                fix = 30
            case 300:
                fix = 80
            case 4000:
                fix = 130
            case 500:
                fix = 185
            case 600:
                fix = 230
        xval = fix - np.array(xval) * 2.5 + 2.5
        print("yerr raw = ", yerr)
        for i in range(len(yerr)):
            yerr[i] = yerr[i]
        print("here come the input values")
        print("xval=",xval)
        print("yval=",yval)
        print("new yerr =", yerr)
        a, b = calculation2.calculation(xval, yval, yerr)
        print("a,b for {} is".format(temp1[0][0]), [a, b])
        new.append([a, b])
        print("**********************************************************")

    print(new)

    xs = np.array([100, 200, 304, 405, 511, 600])
    force = np.array(xs) * 9.80665
    ys = []
    yerrs = []
    for element in new:
        ys.append(element[0])
        yerrs.append(element[1])
    ys = np.array(ys)
    yerrs = np.array(yerrs)
    ys[0] = ys[0] - 360
    c,cerr = cali.cali()
    ys = (ys + 360 - c)/180
    ys = 1 + ys
    yerrs = np.sqrt(yerrs**2+(cerr/c)**2)/180


    """plt.figure()
    plt.errorbar(force, ys, yerr=yerrs)
    plt.xlabel("force")
    plt.ylabel("fringe order")
    plt.show()"""

    result = c3.calculation(force, ys, yerrs)

    print("result=", result)
    """
    writer = np.array(excel_result)
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet('middle_step', cell_overwrite_ok=True)
    sheet2 = book.add_sheet('2', cell_overwrite_ok=True)
    sheet.write(0, 1, "pressure")
    sheet.write(0, 2, "angle count")
    sheet.write(0, 3, "mean")
    sheet.write(0, 4, "standard error")

    # sheet.write(0, 5, "normalised results")
    for i in range(len(writer)):
        for j in range(len(writer[0])):
            sheet.write(i + 1, j + 1, float("{}".format(writer[i][j])))

    for i in range(len(force)):
        sheet2.write(i+1, 2, float("{}".format(force[i])))
    for i in range(len(ys)):
        sheet2.write(i+1, 3, float("{}".format(ys[i])))
    for i in range(len(yerrs)):
        sheet2.write(i+1, 4, float("{}".format(yerrs[i])))
    book.save('middle_step.xls')"""
