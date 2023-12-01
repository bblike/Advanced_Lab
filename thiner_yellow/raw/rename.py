import os
import cv2 as cv

path = r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_yellow\raw"
files = os.listdir(path)

num = [21, 21, 17, 17, 21, 21]
counter = 4291
final = 0

for i in range(len(num)):
    for j in range(num[i]):
        print(r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_yellow\raw\CIMG{}.JPG".format(counter))
        im = cv.imread(r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_yellow\raw\CIMG{}.JPG".format(counter))
        cv.imshow("test", im)
        cv.waitKey(1)
        cv.destroyAllWindows()
        cv.imwrite("{}P{}.JPG".format(i+1, j+1), im)
        print("saved as: {}P{}.JPG".format(i+1, j+1))
        counter += 1
        final = j
    print("group {} end, total number {}.".format(i, final+1))

print("done")