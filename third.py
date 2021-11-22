import cv2
import glob
import os
import math
import argparse

import imutils
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from scipy import ndimage
from skimage import img_as_ubyte, morphology, filters, io, color, exposure
from skimage.feature import canny
from skimage.morphology import binary_dilation, square, disk, diamond, binary_opening


def show_image(image, gray=True, BGR=True):
    out = image.copy()
    if gray == True:
        plt.imshow(image, cmap="gray")
    else:
        if BGR:
            out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(out)
    plt.show()
def calculate_avg_color(image):
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_list = []
    g_list = []
    b_list = []
    for row in image:
        for r, g, b in row:
            if (r != 0 and g != 0 and b != 0):
                r_list.append(r)
                g_list.append(g)
                b_list.append(b)
    r_list = np.sort(r_list)
    g_list = np.sort(g_list)
    b_list = np.sort(b_list)
    avg_r = np.average(r_list[:])
    avg_g = np.average(g_list[:])
    avg_b = np.average(b_list[:])


    return avg_r, avg_g, avg_b

def calculate_average_distance(image):
    distance_list = []
    for row in image:
        for (b, g, r) in row:
            # Only our pixels, not added black background
            if (b != 0 and g != 0 and r != 0):
                # Calculate distance
                value = abs(int(b) - int(g)) + abs(int(b) - int(r)) + abs(int(r) - int(g))

                # Append calculated value
                distance_list.append(value)

    # Cast list to numpy array
    distance_list = np.array(distance_list, dtype="int")

    # Calculate average
    avg = np.average(distance_list)
    return avg
def calculate_average_distance_2(image):
    distance_list = []
    for row in image:
        for (b, g, r) in row:
            # Only our pixels, not added black background
            if (b != 0 and g != 0 and r != 0):
                # Calculate distance
                value = abs(int(b) - int(g)) + abs(int(b) - int(r)) + abs(int(r) - int(g))

                # Append calculated value
                distance_list.append(value)

    # Cast list to numpy array
    distance_list = np.array(distance_list, dtype="int")

    distance_list = np.sort(distance_list)
    # Calculate average

    mini = np.average(distance_list[:100])
    maxi = np.average(distance_list[-100:])
    avg = maxi - mini
    return avg

def make_coin_decision(center_avg, ring_avg):
    if (center_avg < 20.0 or ring_avg < 20.0):
        decision = "Skip image"
        money = 0
    elif (center_avg < 120.0):
        if (ring_avg < 120.0):
            decision = "1 PLN"
            money = 1.00
        else:
            decision = "2 PLN"
            money = 2.00
    else:
        if (ring_avg < 120.0):
            decision = "5 PLN"
            money = 5.00
        else:
            decision = "0.05 PLN"
            money = 0.05

    return decision, money


def make_banknote_decision(avg_color):
    if (avg_color > 95.0):
        decision = "50 PLN"
        money = 50.00
    elif (avg_color > 61.0):
        decision = "10 PLN"
        money = 10.00
    elif (avg_color > 40.0):
        decision = "100 PLN"
        money = 100.00

    else:
        decision = "20 PLN"
        money = 0.00

    return decision, money


const_colors = [(255, 0, 255),  # UNKNOWN
                (0, 255, 0),  # 0.05 PLN
                (255, 0, 0),  # 1 PLN
                (0, 0, 255),  # 2 PLN
                (128, 107, 59),  # 5 PLN
                (114, 97, 68),  # 10 PLN
                (142, 161, 226),  # 50 PLN
                (115, 175, 114),  # 100 PLN
                ]


def find_color(money):
    if (money == 0.05):
        color = const_colors[1]
    elif (money == 1.00):
        color = const_colors[2]
    elif (money == 2.00):
        color = const_colors[3]
    elif (money == 5.00):
        color = const_colors[4]
    elif (money == 10.00):
        color = const_colors[5]
    elif (money == 50.00):
        color = const_colors[6]
    elif (money == 100.00):
        color = const_colors[7]
    else:
        color = const_colors[0]
    return color[::-1]



def intersection_boolean_2(a, b):
    X = a[0]+ a[2]/2
    Y = a[1]+ a[3]/2
    if X < (b[0]+ b[2]) and X > b[0] and Y > b[1 ] and (Y < b[1]+ b[3]):
        return True
    else:
        return False


def find_rectangle(img):
    max_area = (img.shape[0] - 15) * (img.shape[1] - 15)
    print(max_area, 'max area')
    img = cv2.GaussianBlur(img, (15, 15), 0)
    # show_image((img))
    rectangle = []

    for gray in cv2.split(img):
        for thrs in range(0, 255, 5):
            if thrs == 0:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (17, 17), 0)

            ret2, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ret, thresh = cv2.threshold(blur, ret2, 255, cv2.THRESH_BINARY_INV)
            canny = cv2.Canny(thresh, 40, 160)
            dilated = cv2.dilate(canny, (1, 1), iterations=2)

            contours, _hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt in contours:
                flag_intersection = False
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)

                if len(cnt) > 3 and len(cnt) < 15 and cnt.all() != 0 and cv2.contourArea(
                        cnt) > 10000 and cv2.isContourConvex(cnt):

                    x, y, width, height = cv2.boundingRect(cnt)
                    r1 = (x, y, width, height);
                    for r in rectangle:  # for every rectangle in results check that another rectangle intersection with it. If yes then skip it
                        xr, yr, widthr, heightr = cv2.boundingRect(r)
                        r2 = (xr, yr, widthr, heightr);
                        if intersection_boolean_2(r1, r2):
                            flag_intersection = True
                            break

                    if (flag_intersection == False):
                        cnt = cnt.reshape(-1, 2)
                        rectangle.append(cnt)
    print(len(rectangle), 'rects0-=-----------------------------')
    return rectangle

def image_proportion(image):
    x, y, z = image.shape
    if x > y:
        abs_ = abs(x-y)/x
        side = 'y'
    else:
        abs_ = abs(x-y)/y
        side = 'x'
    return abs_, side

def image_rotate(image, n):
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())
    rotated = imutils.rotate_bound(image, n)
    #cv2.imshow("Rotated (Correct)", rotated)
    #cv2.waitKey(0)
    return rotated

def image_resizeing(image, proportion):
    if proportion > 0.3:
        banknote_normalized = cv2.resize(image, (1000, 500))
        banknote_centre = banknote_normalized[125:375, 225:775]
    else:
        banknote_normalized = cv2.resize(image, (1000, 750))

        banknote_centre = banknote_normalized[275:475, 310:690]
    return banknote_centre
if __name__ == '__main__':
    results_dir = "results/"

    # Create new directory when not exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Find files to read
    files_name_list = glob.glob("data/*")

    # Read files
    image_list = list(map(cv2.imread, files_name_list))

    # Iterate on images
    for index, image in enumerate(image_list):
        print(str(files_name_list[index]))
        all_money_list = [0]
        output = image.copy()
        overlay = image.copy()

        rx = None
        ry = None
        rw = None
        rh = None

        # Find banknotes
        banknote_image = image.copy()
        rectangle = find_rectangle(banknote_image)
        for img in rectangle:
            x, y, width, height = cv2.boundingRect(img)
            rx = x
            ry = y
            rw = width
            rh = height
            banknote_to_test = banknote_image[y: y + height, x: x + width].copy()
            proportion, side = image_proportion(banknote_to_test)

            if side == 'y':
                banknote_rotaded = image_rotate(banknote_to_test.copy(), 90)
            else:
                banknote_rotaded = banknote_to_test

            banknote_centre = image_resizeing( banknote_rotaded, proportion)


            #cv2.imshow('one', banknote_centre)
            #cv2.waitKey(0)
            test_avg = calculate_average_distance(banknote_centre)
            print(int(calculate_avg_color(banknote_centre)[2]))
            decision, money = make_banknote_decision(test_avg)

            cv2.rectangle(overlay, (x, y), (x + width, y + height), find_color(money), -1)
            cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
            cv2.rectangle(output, (x, y), (x + width, y + height), find_color(money), 10)
            cv2.putText(output, "{:.2f} PLN".format(money), (int(x + width / 2), int(y + height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (204, 119, 0), 3)
            all_money_list.append(money)

        path = results_dir + files_name_list[index].split('\\')[1]
        cv2.imwrite(path, output)
