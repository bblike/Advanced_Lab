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
import calculation4 as c4
import calculation5 as c5
#


path = r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_green"
files = os.listdir(path)
# path = "newtest.JPG"
file_length = len(files)
selector = [1, 2, 3, 4, 5, 6, 7]
excel_result = []
diameter = [69, 0.006]
thickness = [9.590,0.002]
lamb = [545, 50]
centre = []

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
        for i in selector:
            for j in range(0, 30):
                if file == "{}P{}.JPG".format(i, j):

                    im = cv.imread(r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_green\{}".format(file), 0)
                    #print(np.shape(im))
                    circles = cv.HoughCircles(im, cv2.HOUGH_GRADIENT,1, 200, param2=20, minRadius=200, maxRadius=400)
                    arr1 = np.zeros([0, 2], dtype=int)
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for n in circles[0, :]:
                            arr1 = np.append(arr1, [n[0], n[1]])

                    # select the centre area
                    #print("begin")
                    #print("arr1=", arr1)
                    selected = list([arr1[0], arr1[1]])
                    #print(selected)
                    centre.append(selected)
                    #print(np.shape(selected))

                    for m in range(2, len(arr1), 2):
                        arg1 = np.sqrt((2816/2-arr1[m])**2+(2108/2 -arr1[m+1])**2)
                        arg2 = np.sqrt((2816/2-selected[0])**2+(2108/2 -selected[1])**2)
                        if arg1 < arg2:
                            selected = np.array([arr1[m], arr1[m+1]])
                    #print("selected=",selected)
                    #print(np.shape(selected))
                    assert np.shape(selected) == (2,)

                    r = 23
                    new_array = im[selected[1] - r:selected[1] + r, selected[0] - r:selected[0] + r]
                    # im2 = cv.imread(r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_green\{}".format(file))
                    """
                    for o in np.arange(selected[1] - r, selected[1] + r):
                        for p in np.arange(selected[0] - r, selected[0] + r):
                            im2[o][p] = [255, 0, 0]
                    cv.namedWindow("test", 0)
                    cv.imshow("test", im2)

                    cv.waitKey(1000)
                    cv.destroyAllWindows()
                    #print(new_array)"""

                    r_circle = 20   # radius of the circle selected
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

    #groups = [[group600]]
    groups = [[group100], [group200], [group300], [group400], [group500], [group600], [group700]]
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
            case 1:
                fix = 290
            case 2:
                fix = 330
            case 3:
                fix = 380
            case 4:
                fix = 410
            case 5:
                fix = 450
            case 6:
                fix = 510
            case 7:
                fix = 550
        xval = fix - np.array(xval) * 2.5 + 2.5

        print("here come the input values")
        print("xval=",xval)
        print("yval=",yval)
        print("new yerr =", yerr)

        a, b = c5.calculation(xval, yval, yerr, temp1[0][0])
        print("a,b for {} is".format(temp1[0][0]), [a, b])
        new.append([a, b])
        print("**********************************************************")

    print(new)

    xs = np.array([109.3, 207.0, 295.2, 402.3, 481.9, 545.9, 614.7])
    force = np.array(xs) * 9.80665
    ys = []
    yerrs = []
    for element in new:
        ys.append(element[0])
        yerrs.append(element[1])
    ys = np.array(ys)
    print("ys=", ys)
    yerrs = np.array(yerrs)
    c,cerr = 5, 0
    ys = (ys - c)/180
    ys = 1 + ys
    yerrs = np.sqrt(yerrs**2+(cerr/c)**2)/180


    """plt.figure()
    plt.errorbar(force, ys, yerr=yerrs)
    plt.xlabel("force")
    plt.ylabel("fringe order")
    plt.show()"""

    result = c4.calculation(force, ys, yerrs, diameter, thickness, lamb)

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
