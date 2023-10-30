# Numpy, Matplotlib and Pillow was used. Remember to import.
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp
import datetime
import math
import time
import rgb2gray
import interactive as inte
from PIL import Image
import threading as td
from queue import Queue
#



path = "CIMG3813.JPG"

#global im
im = cv.imread(path)
#global structure
structure = np.shape(im)
print(structure)

#global red_structure
red_structure = np.array([structure[0], structure[1], 3])
#global red
red = np.zeros(red_structure)



result = []
#calculation_red()




def calculation_red(q):
    for i in range(0, red_structure[0]):
        for j in range(0, red_structure[1]):

            red[i][j][1] = 150

        print("process{} done.".format(i))
    q.put(1)



def calculation_red_param(param):

    for i in param:
        for j in range(0, red_structure[1]):

            red[i][j][0] = 150


        print("process {} done.".format(i))
    print("calculation {} down".format(param))
    print("test: ", red[(param[0])][100][0])
def parallel():
    begin_time = time.time()
    n = structure[0]
    procs = []
    n_cpu = int(mp.cpu_count()/2)
    chunk_size = int(n/n_cpu)
    x = 0
    for i in range(0, n_cpu):
        min_i = chunk_size * i

        if i < n_cpu-1:
            max_i = chunk_size * (i+1)
        else:
            max_i = n
        digits = []
        for digit in range(min_i, max_i):
            digits.append(digit)
        print(digits)
        print(len(digits))
        procs.append(mp.Process(target=calculation_red_param, args = ([digits])))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    end_time = time.time()
    print("time spend", end_time-begin_time)


def threading_job():
    q = Queue()
    ts = []
    begin_time = time.time()
    for i in range(0, 4):
        t =mp.Process(target = calculation_red, args = (q,))
        t.start()
        ts.append(t)

    for i in ts:
        i.join()
    results = []
    for i in range(0,4):
        results.append(q.get())
    end_time = time.time()
    print("all done")
    print("results=", results)
    print("time_spent=", end_time-begin_time)



if __name__ == '__main__':

    parallel()
    #threading_job()
    print("waiting...")
    cv.waitKey(3000)
    cv.imshow("new window", red)
    cv.waitKey(2000)
    cv.destroyAllWindows()
    cv.imwrite("raw-r.jpg", red)






