# Numpy, Matplotlib and Pillow was used. Remember to import.
import cv2
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
import calculation4
#


path = r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\stair_structure_research\test"
files = os.listdir(path)
# path = "newtest.JPG"
file_length = len(files)
selector = [1, 2, 3]
excel_result = []
poi = []
selected = [1120, 1100]


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
            for j in range(0, 30):
                if file == "{}P{}.JPG".format(i, j):

                    im = cv.imread(r"test\{}".format(file))
                    """print(im.shape)
                    print(im[selected[0]][selected[1]])
                    for n in range(selected[0]-5, selected[0]+5):
                        for m in range(selected[1]-5, selected[1]+5):
                            im[n][m][0] = 255
                            im[n][m][1] = 0
                            im[n][m][2] = 0
                    print(im[selected[0]][selected[1]])
                    cv.namedWindow("test", 0)
                    cv.imshow("test", im)
                    cv.waitKey(10000)
                    cv.destroyAllWindows()
                    """

                    r = 20
                    new_array = im[selected[0] - r:selected[0] + r, selected[1] - r:selected[1] + r]
                    #print(new_array)
                    r_circle = 3   # radius of the circle selected
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
    group700 = []

    for element in excel_result:
        match element[0]:
            case 1:
                group100.append(element)
            case 2:
                group200.append(element)
            case 3:
                group300.append(element)
            case 4:
                group400.append(element)
            case 5:
                group500.append(element)
            case 6:
                group600.append(element)
            case 7:
                group700.append(element)

    groups = [[group100], [group200], [group300]]
    #groups = [[group100], [group200], [group300], [group400], [group500], [group600], [group700]]
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
        match temp[0]:
            case 1:
                fix = 130
            case 2:
                fix = 130
            case 3:
                fix = 130
            case 4:
                fix = 420
            case 5:
                fix = 460
            case 6:
                fix = 520
            case 7:
                fix = 570
        xval = fix + np.array(xval) * 2.5 - 2.5

        print("here come the input values")
        print("xval=",xval)
        print("yval=",yval)
        print("new yerr =", yerr)

        a, b = calculation2.calculation(xval, yval, yerr, temp[0])
        print("a,b for {} is".format(temp[0][0][0]), [a, b])
        new.append([a, b])
        print("**********************************************************")

    print(new)

    xs = np.array([100, 200, 300])
    force = np.array(xs) * 9.80665
    ys = []
    yerrs = []
    for element in new:
        ys.append(element[0])
        yerrs.append(element[1])
    ys = np.array(ys)
    yerrs = np.array(yerrs)
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
