import os
import cv2 as cv
import interactive as inte

path = r"C:\Users\{}\Desktop\Advanced_Lab\last_groups\large_size\raw".format(inte.name)
files = os.listdir(path)

num = [21, 17, 17, 17, 17, 17, 17]
counter = 4547
final = 0

for i in range(len(num)):
    for j in range(num[i]):
        print(r"C:\Users\{}\Desktop\Advanced_Lab\last_groups\large_size\raw\CIMG{}.JPG".format(inte.name, counter))
        im = cv.imread(r"C:\Users\{}\Desktop\Advanced_Lab\last_groups\large_size\raw\CIMG{}.JPG".format(inte.name, counter))
        #cv.imshow("test", im)
        #cv.waitKey(1)
        #cv.destroyAllWindows()
        cv.imwrite("{}P{}.JPG".format(i+1, j+1), im)
        print("saved as: {}P{}.JPG".format(i+1, j+1))
        counter += 1
        final = j
    print("group {} end, total number {}.".format(i, final+1))

print("done")