import os
import cv2 as cv

path = r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_green\raw"
files = os.listdir(path)

num = [17, 21, 25, 21, 21, 25, 21]
counter = 4138
final = 0

for i in range(len(num)):
    for j in range(num[i]):
        print(r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_green\raw\CIMG{}.JPG".format(counter))
        im = cv.imread(r"C:\Users\Li Zhejun\Desktop\Advanced_Lab\thiner_green\raw\CIMG{}.JPG".format(counter))
        cv.imshow("test", im)
        cv.waitKey(1)
        cv.destroyAllWindows()
        cv.imwrite("{}P{}.JPG".format(i+1, j+1), im)
        print("saved as: {}P{}.JPG".format(i+1, j+1))
        counter += 1
        final = j
    print("group {} end, total number {}.".format(i, final+1))

print("done")