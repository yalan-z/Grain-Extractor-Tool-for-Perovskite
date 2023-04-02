import cv2
from skimage import data, measure, morphology
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly
import plotly.express as px
import plotly.graph_objects as go
import time
import pdb
import math

impath = 'test_image'  # image filepath
imname = 'resulttest.png'   #image name
rgb = np.loadtxt('params/rgbcolor.txt')   #RGB color template
ratio=4    #SEM magnitude (unit:10k)
image = cv2.imread(impath + '\\' + imname)
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, imthresh = cv2.threshold(imgray, 253, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def moropen(bimage, k1, k2, numi):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, k1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k2)
    erosion = cv2.erode(bimage, kernel1, iterations=numi)
    imageopen = cv2.dilate(erosion, kernel2, iterations=3)
    return imageopen


def openlarge(bimage, k5, k6, numi):
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k5)
    kernel6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k6)
    erosion = cv2.erode(bimage, kernel5, iterations=numi)
    imageopen = cv2.dilate(erosion, kernel6, iterations=1)
    return imageopen


def largeerosion(bimage, k7, k8, numi):
    kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, k7)
    kernel8 = cv2.getStructuringElement(cv2.MORPH_CROSS, k8)
    erosion = cv2.erode(bimage, kernel7, iterations=numi)
    imageopen = cv2.dilate(erosion, kernel8, iterations=1)
    return imageopen


def close(bimage, k3, k4, numi):
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k3)
    kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS, k4)
    dilation = cv2.dilate(bimage, kernel3, iterations=numi)
    imageclose = cv2.erode(dilation, kernel4, iterations=1)
    return imageclose


def visualabels(image, num_labels, labels, savename):
    labelplot = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        j = random.randrange(0, len(rgb) - 1)
        labelplot[:, :, 0][mask] = rgb[j][2]
        labelplot[:, :, 1][mask] = rgb[j][1]
        labelplot[:, :, 2][mask] = rgb[j][0]
    cv2.imshow('label', labelplot)
    cv2.waitKey()
    cv2.imwrite(impath + '\\' + savename + '.jpg', labelplot)
    cv2.imwrite(impath + '\\' + savename + '.png', labels)
    return


def distance(centerp, regionp):
    dpoint = regionp - centerp
    d = np.sqrt(np.sum(np.square(dpoint)))
    return d


def labelcontour(label, num_labels):
    arraycon = np.empty([num_labels - 1, 1])
    labelarray = np.zeros((image.shape[0], image.shape[1]), np.float_)
    for i in range(1, num_labels):
        mask = label == i
        labelarray[:, :][mask] = 255
        contours = measure.find_contours(labelarray)
        arraycon[i - 1] = contours
    return arraycon

def GSAcalculation(count,txtname):
    width = 1 / 28.346 / ratio #pixel width in real space
    pixelarea = width ** 2 
    area =count * pixelarea
    areal=area.tolist()
    pathper = open(impath + '\\' + txtname, 'w', encoding='utf-8')  # txt file path
    for x in areal:
        print(x, file=pathper)
    return



lightopen = moropen(imthresh, (4, 4), (2, 2), 3)
afteropen = openlarge(imthresh, (4, 4), (1, 1), 4)
numex, labelextract, statex, centroidfalt = cv2.connectedComponentsWithStats(lightopen, connectivity=4)
numl, labelatlarge, statero, centroids = cv2.connectedComponentsWithStats(afteropen, connectivity=4)

visualabels(image, numex, labelextract, 'name1')
countex = statex[:, 4]
GSAcalculation(countex,'test.txt')
visualabels(image, numl, labelatlarge, 'name2')
propsero = measure.regionprops(labelatlarge)
propex = measure.regionprops(labelextract)
properties = 'centroid'
n = 1
labelmax = labelextract.max()
for j in range(1, labelmax + 1):
    listcen = []
    labelarray = np.zeros((image.shape[0], image.shape[1]), np.float_)
    mask = labelextract == j
    labelarray[:, :][mask] = 255
    contours = measure.find_contours(labelarray)[0]

    for i in range(0, labelatlarge.max() - 1):
        points = np.array([[propsero[i].centroid[0], propsero[i].centroid[1]]])
        if measure.points_in_poly(points, contours):
            listcen.append(propsero[i].centroid)
    if len(listcen) == 0:
        print('listcen=0 at j=' + str(j))
    if len(listcen) == 1:
        continue
    if len(listcen) == 2:
        for i2 in range(0, propex[j - 1].area):
            d1 = distance(np.array(listcen[0]), propex[j - 1].coords[i2])
            d2 = distance(np.array(listcen[1]), propex[j - 1].coords[i2])
            if d1 > d2:
                labelextract[propex[j - 1].coords[i2][0], propex[j - 1].coords[i2][1]] = n + labelmax
        n += 1
    if len(listcen) == 3:
        m = n + 1
        for i2 in range(0, propex[j - 1].area):
            d1 = distance(np.array(listcen[0]), propex[j - 1].coords[i2])
            d2 = distance(np.array(listcen[1]), propex[j - 1].coords[i2])
            d3 = distance(np.array(listcen[2]), propex[j - 1].coords[i2])
            if d1 > d2 and d2 > d3:
                continue
            if d1 > d2 and d3 > d2:
                labelextract[propex[j - 1].coords[i2][0], propex[j - 1].coords[i2][1]] = n + labelmax
            if d1 < d2 and d3 > d1:
                labelextract[propex[j - 1].coords[i2][0], propex[j - 1].coords[i2][1]] = m + labelmax
            if d1 < d2 and d3 < d1:
                continue
        n = m + 1

visualabels(image, numex + n - 1, labelextract, 'final')

countf=[]
for l in range(0,numex + n - 1):
    countf.append(np.sum(labelextract==l))
countf=np.array(countf,dtype=int)
GSAcalculation(countf,'test2.txt')