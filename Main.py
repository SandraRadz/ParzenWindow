import math
import random

import pylab as pl
import scipy as sp
from matplotlib.colors import ListedColormap


def generateData(classNum, pointInClassNum, spread, sizex, sizey):
    data = []
    for cN in range(classNum):
        centerX = random.random() * sizex
        centerY = random.random() * sizey
        for el in range(pointInClassNum):
            data.append([[random.gauss(centerX, spread), random.gauss(centerY, spread)], cN])
    return data


def splitData(data):
    trainData = []
    testData = []
    for point in data:
        if random.random() < 0.7:
            trainData.append(point)
        else:
            testData.append(point)
    return trainData, testData


def showData(traindata, colors):
    classColormap = ListedColormap(colors)
    pl.scatter([traindata[i][0][0] for i in range(len(traindata))],
               [traindata[i][0][1] for i in range(len(traindata))],
               c=[traindata[i][1] for i in range(len(traindata))],
               cmap=classColormap)


# a, b =[x, y]
def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def findNearestPoints(point, data, windowsize):
    nearestpoint = []
    for i in data:
        d = dist(point, i[0])
        if d <= windowsize:
            nearestpoint.append([d, i[1]])
    return sorted(nearestpoint)


def maxIndex(data):
    if (len(data)) == 0:
        return -1
    max = data[0]
    maxIndex = 0
    for i in range(len(data)):
        if data[i] > max:
            maxIndex = i
            max = data[i]
    return maxIndex


# testData without class
def calculateClass(trainData, testData, classNum, windowsize, coreType):
    testWithLabels = []
    for i in range(len(testData)):
        stat = [0 for i in range(classNum)]
        nearestpoints = findNearestPoints(testData[i], trainData, windowsize)
        if coreType==0:
            for j in range(len(nearestpoints)):
                stat[nearestpoints[j][1]] += 1
        elif coreType == 1:
            for j in range(len(nearestpoints)):
                stat[nearestpoints[j][1]] += (1-nearestpoints[j][0]/windowsize)
        elif coreType == 2:
            for j in range(len(nearestpoints)):
                stat[nearestpoints[j][1]] += (1-(nearestpoints[j][0]/windowsize)**2)
        elif coreType == 3:
            for j in range(len(nearestpoints)):
                stat[nearestpoints[j][1]] += ((1-(nearestpoints[j][0]/windowsize)**2)**2)
        else:
            for j in range(len(nearestpoints)):
                e=2.71828
                stat[nearestpoints[j][1]] += e**(-2*(nearestpoints[j][0]/windowsize)**2)
        testWithLabels.append([testData[i], maxIndex(stat)])
    return testWithLabels


# with labels
def calculateAccuracy(testData, myTestData):
    return sum([int(testData[i][1] == myTestData[i][1]) for i in range(len(myTestData))]) / float(len(myTestData))


def trainWindowSize(trainData, testDataL, testData, classNum, maxsize, method):
    res = []
    variants = sp.linspace(0.5, maxsize, 10)
    for i in range(len(variants)):
        res.append(calculateAccuracy(testDataL, calculateClass(trainData, testData, classNum, i, method)))
    q = maxIndex(res)
    return [variants[q], res[q]]


_classNum = 3
_pointNum = 40
_spread = 0.5
_sizex = 3
_sizey = 3
_windowSize = 1.5
_maxsize = _sizex
_method = 4
methods=["rectangular core", "triangular core", "square core", "super square core", "Gauss core"]

# method 0 - rectangular core
# method 1 - triangular core
# method 2 - square core
# method 3 - super square core
# method 4 - Gauss core


# create data
data = generateData(_classNum, _pointNum, _spread, _sizex, _sizey)
trainData, testWithLabel = splitData(data)
testData = [[testWithLabel[i][0][0], testWithLabel[i][0][1]] for i in range(len(testWithLabel))]
# train window size
res = []
reshelper = []
for i in range(5):
    tws = trainWindowSize(trainData, testWithLabel, testData, _classNum, _maxsize, i)
    res.append([tws[0], tws[1], i])
    reshelper.append(tws[1])
maxIndex(reshelper)
# find the best size
windowSize = res[maxIndex(reshelper)]
# create new array with label
pSize = res[maxIndex(reshelper)][0]
acc = res[maxIndex(reshelper)][1]
mNum = res[maxIndex(reshelper)][2]
res1 = calculateClass(trainData, testData, _classNum, pSize, mNum)
print("best window size:", pSize)
print("best method:", methods[mNum])
print("accuracy:", acc)
# draw
colors1 = ['#CD5555', '#104E8B', '#008B00']
colors2 = ['#FFC1C1', '#BBFFFF', '#9AFF9A']
showData(trainData, colors1)
showData(res1, colors2)

#res2 = calculateClass(trainData, [[2, 3], [0, 1], [3, 4], [0, 4]], _classNum, 1.5, 1)
#showData(res2, colors2)
pl.show()
