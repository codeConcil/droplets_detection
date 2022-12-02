import os

from PIL import Image
import cv2
import random
import numpy as np
from skimage.util import random_noise
import scipy.ndimage as ndi

global_b = 0
global_bp = 0


def find_unchor(circle):
    left = temp[0]
    up = temp[0]
    right = temp[0]
    down = temp[0]
    for counter in temp:
        if counter[0] < left[0]:
            left = counter
        if counter[0] > right[0]:
            right = counter
        if counter[1] < up[1]:
            up = counter
        if counter[1] > down[1]:
            down = counter
    direction = [up, down, left, right]

    return direction


def get_yolo(circle):
    global global_b
    global global_bp
    p_1 = [circle[2][0], circle[0][1]]
    p_2 = [circle[3][0], circle[0][1]]
    p_3 = [circle[2][0], circle[1][1]]
    p_4 = [circle[3][0], circle[1][1]]
    width = (p_2[0] - p_1[0]) / im.size[0]
    height = (p_4[1] - p_2[1]) / im.size[1]
    width_p = p_2[0] - p_1[0]
    height_p = p_4[1] - p_2[1]
    if width_p / height_p > 1.2:
        yolo_format = [1, ((p_2[0] + p_1[0]) / 2) / im.size[0], ((p_4[1] + p_2[1]) / 2) / im.size[1], width, height]
        global_bp += 1
    else:
        yolo_format = [0, ((p_2[0] + p_1[0]) / 2) / im.size[0], ((p_4[1] + p_2[1]) / 2) / im.size[1], width, height]
        global_b += 1
    return yolo_format


def add_modifier(image_name, folder_i, folder_dir):
    img = cv2.imread(os.path.join(folder_dir, image_name))

    random_field = 20
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = False
    noise = False
    if random_field == 0:
        mn = random.uniform(0.01, 0.05)
        vr = random.uniform(0.01, 0.05)
        noise_img = random_noise(img, mode='gaussian', mean=mn, var=vr)
        noise = True
    elif random_field == 1:
        sz = random.randint(5, 12)
        blurred = ndi.uniform_filter(img, size=sz, mode='reflect')
        blur = True
    elif random_field == 2:
        mn = random.uniform(0.01, 0.2)
        vr = random.uniform(0.01, 0.5)
        noise_img = random_noise(img, mode='speckle', mean=mn, var=vr)
        noise = True
    elif random_field == 3:
        sz = random.randint(5, 12)
        blurred = ndi.uniform_filter(img, size=sz, mode='constant')
        blur = True
    elif random_field == 4:
        noise_img = random_noise(img, mode='poisson')
        noise = True
    elif random_field == 5:
        sz = random.randint(5, 12)
        blurred = ndi.uniform_filter(img, size=sz, mode='nearest')
        blur = True
    elif random_field == 6:
        am = random.uniform(0.01, 0.1)
        noise_img = random_noise(img, mode='salt', amount=am)
        noise = True
    elif random_field == 7:
        sz = random.randint(5, 12)
        blurred = ndi.uniform_filter(img, size=sz, mode='mirror')
        blur = True
    elif random_field == 8:
        am = random.uniform(0.01, 0.5)
        noise_img = random_noise(img, mode='pepper', amount=am)
        noise = True
    elif random_field == 8:
        sz = random.randint(5, 12)
        blurred = ndi.uniform_filter(img, size=sz, mode='wrap')
        blur = True
    elif random_field == 9:
        am = random.uniform(0.01, 0.3)
        sv = random.uniform(0.01, 0.3)
        noise_img = random_noise(img, mode='s&p', amount=am, salt_vs_pepper=sv)
        noise = True

    if noise:
        noise_img = np.array(255 * noise_img, dtype='uint8')
        # cv2.imshow('blur', noise_img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(folder_i, image_name), noise_img)
    elif blur:
        cv2.imwrite(os.path.join(folder_i, image_name), blurred)
        # cv2.imshow('blur', blurred)
        # cv2.waitKey(0)

#Path to folder with generated mask images
#folder_dir =
#Example
#folder_dir = 'C:\Blobs\Blobs_color\second_set\masks'


#----------------------
#Not used in last version
#folder_i = 'C:\Blobs\images1'
#--------------------------

#Path to folder, when you want save labels. Usually, beside folder of color images
#folder_t =
#Example
#folder_t = 'C:\Blobs\Blobs_color\second_set\labels'

for filename in os.listdir(folder_dir):
    im = Image.open(os.path.join(folder_dir, filename))
    im = im.convert('HSV')
    pix = im.load()
    temp = []
    # print(pix[800, 409])
    circles = 0
    points = 0
    dir = []
    for i in range(0, im.size[0]):
        for j in range(0, im.size[1]):
            color = pix[i, j]

            # print(color)
            #color[1] < 200 and
            # if 120 < color[0] < 240 and color[1] > 100 and 40 < color[2] < 90:
            if color[2] > 10:
                points += 1
                if len(temp) == 0:
                    circles = 1
                # print(i, j)
                # color = (255, 0, 0)
                temp.append([i, j])
            # pix[i, j] = color
        if points == 0 and circles == 1:
            dir.append(find_unchor(temp))
            circles = 0
            temp.clear()
        points = 0

    for i in range(0, im.size[0]):
        for j in range(0, im.size[1]):
            if j == dir[0][0][1] and dir[0][2][0] <= i <= dir[0][3][0]:
                pix[i, j] = (255, 0, 0)
        # if j == down[1] and left[0] <= i <= right[0]:
        #     pix[i, j] = (255, 0, 0)
        # if i == left[0] and up[1] <= j <= down[1]:
        #     pix[i, j] = (255, 0, 0)
        # if i == right[0] and up[1] <= j <= down[1]:
        #     pix[i, j] = (255, 0, 0)

    yolo_form = []
    for circle in dir:
        yolo_form.append(get_yolo(circle))

    # num_yolo = numpy.array(yolo_form)
    # numpy.savetxt('test.txt', num_yolo, delimiter=' ', fmt='%f')

    f = open(os.path.join(folder_t, os.path.splitext(filename)[0]) + '.txt', "w")
    #print(os.path.join(folder_t, filename[0]))
    for item in yolo_form:
        i = 0
        for counter in item:
            if i < 4:
                f.write(str(counter) + ' ')
            else:
                f.write(str(counter))
            i += 1
        f.write("\n")

    f.close()
    print(filename + ' processed')
    im.close()
    #add_modifier(filename, folder_i, folder_dir)
print("Blob class count")
print(global_b)
print("Blob+ class count")
print(global_bp)
