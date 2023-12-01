import os
import cv2 as cv
import interactive as inte

path = r"C:\Users\{}\Desktop\Advanced_Lab\last_groups\small_size\raw".format(inte.name)
files = os.listdir(path)

num = [17, 21, 17, 17, 21, 21]
counter = 4670
final = 0

for i in range(len(num)):
    for j in range(num[i]):
        print(r"C:\Users\{}\Desktop\Advanced_Lab\last_groups\small_size\raw\CIMG{}.JPG".format(inte.name, counter))
        im = cv.imread(r"C:\Users\{}\Desktop\Advanced_Lab\last_groups\small_size\raw\CIMG{}.JPG".format(inte.name, counter))
        cv.imwrite("{}P{}.JPG".format(i+1, j+1), im)
        print("saved as: {}P{}.JPG".format(i+1, j+1))
        counter += 1
        final = j
    print("group {} end, total number {}.".format(i, final+1))

print("done")