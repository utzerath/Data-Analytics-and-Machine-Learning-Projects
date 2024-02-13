# Packages
import operator
#Jack Utzerath



from numpy import reshape
from venv import create


import cv2
import pyautogui as pg
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def recreate_image(colors ,labels, w, h,d):
    image = np.zeros((w,h,d))
    label_idx = 0


    #label each pixels according to the limited labels
    for i in range(w):
        for j in range(h):
            image[i][j] = colors[labels[label_idx]]

            label_idx += 1

    return (image)

def image_compression(img, num_clusters):
    #we are going to be using k mean clustering here.
    #basically the algorithm is going to group the similar colors

    image = plt.imread(img)


    type(image)

    print(image.shape)

    print(image.size)

    w,h,d = image.shape

    #joing w and h together
    image_array = image.reshape(w*h, d)

    print(image_array.shape)

    #maximum intensity value
    image_array = image_array/255

    #extracting the small subs and predicting for the whole image

    image_array_sample = shuffle(image_array,random_state=1)[:1000]
    image_array_sample.size

    #introduce kmeans
    kmeans = KMeans(n_clusters = num_clusters, random_state = 1)
    kmeans.fit(image_array_sample)

    #predict
    labels = kmeans.predict(image_array)
    print(kmeans.cluster_centers_)

    #get colors
    colors = kmeans.cluster_centers_

    #recreate orginal image according to labels and each pixels
    image2 = recreate_image(colors, labels, w, h, d)

    #shows the orginal
    plt.figure(1)
    plt.axis('off')
    plt.title('og')
    plt.imshow(image)
    plt.show

    #shows the reduced
    plt.figure(2)
    plt.axis('off')
    plt.title("reduced")
    plt.imshow(image2)
    plt.show()

def print_draw(blackAndWhiteImage):

    time.sleep(3)

    x_start = 500
    y_start = 500

    pg.moveTo(x_start, y_start)
    # loop over pixel rows
    for y in range(len(blackAndWhiteImage)):
        #init row
        row = blackAndWhiteImage[y]

        for x in range(len(row)):
            if row[x] == 0:
                #draw pixel!
                pg.click(x_start + x, y_start + y, _pause = False)
                #print('Drawing at: ', x_start + x, y_start + y)

                #animation speed
                time.sleep(0.001)


def resize(image, amount):

    scale_percent = amount  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    #get the resized image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resize

def process_img(img):

    # convert image to grayScale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #invert the image
    invert_img = cv2.bitwise_not(grayImage)

    #blur the image for better processing
    blur_img = cv2.GaussianBlur(invert_img, (1, 1), 0)

    inverted_image = cv2.bitwise_not(blur_img)

    #show blur image
    cv2.imshow('sketch image', blur_img)

    #convert into threshold
    r, res = cv2.threshold(inverted_image, 108, 255, cv2.THRESH_BINARY)

    return res

if __name__ == '__main__':

    num_clusters = 6

    img =  'C:\\Users\jacku\Downloads\IMG_0025.JPG'

    #read in an image
    image = cv2.imread(img)

    #show image
    cv2.imshow('Og image', image)


    #process the image
    blackAndWhiteImage = process_img(image)


    image_compression(img, num_clusters)





