'''
Created 11/22/2016
@authort Caleb Hulbert
'''
from math import sqrt

from sklearn.cluster import KMeans, DBSCAN

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Kernel(object):
    '''
    Built from a list of pixels(3 tuples), of the form (r,g,b) and converted to pixels of the
    form (l,a,b) where l,a,b are floats which make up a single hunter Lab color.
    Also contains statistical measure of the pixel list.

    clustertype options = ["kmeans","dbscan","stats", "none"]

     Attributes:
        -name               string
        -pixellist          list of [l,a,b] lists
        -numberofpixels     integer
        -clusters           list of clusters of form [size of cluster, cluster mean]
        -mean               [l,a,b]
        -mode               [l,a,b]
        -sd                 [lsd,asd,bsd] (standard deviations)
    '''

    def __init__(self, pixellist, name, clustertype="none", stats=False):
        self.name = name
        self.clustertype = clustertype
        self.pixellist = []
        self.numberofpixels = 0
        self.mode = [0, 0, 0]
        self.mean = [0, 0, 0]
        self.sd = [0, 0, 0]
        self.clusters = []
        self.setpixels(pixellist)
        if stats == True:
            # uncomment if you want more than just the straight mean of all the pixels
            # self.setstats()
            pass
        else:
            if self.clustertype == "kmeans":
                self.clusters = []
                self.kmeans()
                self.mode = [0, 0, 0]
                self.mean = [0, 0, 0]
                self.sd = [0, 0, 0]
            if self.clustertype == "dbscan":
                self.db = self.dbscan()
            if self.clustertype == "none":
                pass

    def setpixels(self, pixellist, calculatemean = False):
        '''
        converts pixels from rgb to hunter lab and sets numberofpixels
        '''
        L = 0
        a = 0
        b = 0
        numberofpixels = 0
        for pixel in pixellist:
            hlab = RGBtoHunterLab(pixel[0], pixel[1], pixel[2])
            L += hlab["L"]
            a += hlab["A"]
            b += hlab["B"]
            self.pixellist.append([hlab["L"], hlab["A"], hlab["B"]])
            numberofpixels += 1
        self.numberofpixels = numberofpixels
        self.mean = [float(L)/numberofpixels, float(a)/numberofpixels, float(b)/numberofpixels]

    def setstats(self):
        '''
        calculates the mean, mode, and sd of the l, a, and b values in the pixellist.
        '''
        frequencydict = {}
        llist, alist, blist = [], [], []
        L = 0
        a = 0
        b = 0
        for pixel in self.pixellist:
            llist.append(pixel[0])
            alist.append(pixel[1])
            blist.append(pixel[2])
            L += pixel[0]
            a += pixel[1]
            b += pixel[2]
            color = "l%.4f,a%.4f,b%.4f" % (pixel[0], pixel[1], pixel[2])
            if color in frequencydict.keys():
                frequencydict[color] += 1
            else:
                frequencydict[color] = 1
        mode = max(frequencydict, key=frequencydict.get)
        self.mode[0] = float(mode[1:mode.index("a") - 1])
        self.mode[1] = float(mode[mode.index("a") + 1:mode.index("b") - 1])
        self.mode[2] = float(mode[mode.index("b") + 1:])
        self.mean[0] = meanstdv(llist)[0]
        self.mean[1] = meanstdv(alist)[0]
        self.mean[2] = meanstdv(blist)[0]
        self.sd[0] = meanstdv(llist)[1]
        self.sd[1] = meanstdv(alist)[1]
        self.sd[2] = meanstdv(blist)[1]

    def kmeans(self, default='mode'):
        '''
        calculates clusets centers based on a k-means algorithm with k=2.
        also determines the size of the clusters, and depending on how balanced each cluster is,
        may default to setting both clusters to the mode or mean.
        '''
        kmeans = KMeans(n_clusters=2).fit(self.pixellist)
        sizec1 = 0
        sizec2 = 0
        for label in kmeans.labels_:
            if label == 0:
                sizec1 += 1
            elif label == 1:
                sizec2 += 1
        largestclustersize = max(sizec1, sizec2)
        smallestclustersize = min(sizec1, sizec2)
        largepercentofsmall = float(
            largestclustersize) / float(smallestclustersize)
        if largepercentofsmall > 1.7:  # TODO: remove magic number
            self.setstats()
            if default == 'mode':
                default = self.mode
            elif default == 'mean':
                default = self.mean
            self.clusters.append([0, default])
            self.clusters.append([0, default])
        else:
            self.cluster.append([sizec1, kmeans.cluster_centers_[0].tolist()])
            self.cluster.append([sizec2, kmeans.cluster_centers_[1].tolist()])

    def dbscan(self, eps=1.5, plot=False):
        self.clusters = []
        X = np.array(self.pixellist)
        db = DBSCAN(eps=eps).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        unique_labels = set(labels)
        colors = []
        for k in unique_labels:
            class_member_mask = (labels == k)
            xyz = X[class_member_mask & core_samples_mask]
            llist, alist, blist = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            if len(llist) > self.numberofpixels / 100:
                lmean = llist.mean()
                amean = alist.mean()
                bmean = blist.mean()
                self.clusters.append([len(llist), [lmean, amean, bmean]])
                r, g, b = HunterLabToRGB(lmean, amean, bmean)
                from random import random
                r = random()*255
                g= random()*255
                b = random()*255
                colors.append([r / 255.0, g / 255.0, b / 255.0])
            else:
                colors.append('k')
        if plot is True:
            graph = plt.figure()
            ax = graph.add_subplot(111, projection='3d')
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = 'k'
                class_member_mask = (labels == k)
                xyz = X[class_member_mask & core_samples_mask]
                if len(xyz[:, 0]) >= self.numberofpixels / 100:
                    print len(xyz[:, 0])
                    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col)
                    xyz = X[class_member_mask & ~core_samples_mask]
                    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[
                        :, 2], c=col, marker='.')
            ax.set_xlabel('L')
            ax.set_ylabel('a')
            ax.set_zlabel('b')
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.ion()
            plt.show()
        return db

    def showscatterplot(self):
        '''
        creates a 3d scatter plot with the points as the [L,a,b] values of the pixels
        this kernel contains. if the mean and the clustser were calculated, they will
        be added as well.
        '''
        lablists = threeTupleToThreeLists(self.pixellist)
        plot = plt.figure()
        plt.close(1)
        del plot
        plot = plt.figure()
        ax = plot.add_subplot(111, projection='3d')
        ax.scatter(lablists[0], lablists[1], lablists[2], c='b', marker='.')
        ax.set_xlabel('L')
        ax.set_ylabel('a')
        ax.set_zlabel('b')
        if self.clusters[0][1] != self.clusters[1][1]:
            addpoints([self.clusters[0][1], self.clusters[1][1]],
                      ax, marker='o')
        elif self.mean != [0, 0, 0]:
            addpoints(self.mean, ax, color='g', marker='o')
        plt.ion()
        plt.show()
        return ax


def threeTupleToThreeLists(threetuple):
    '''
    inputs [[1,1,1],[2,2,2],[3,3,3],[4,4,4]] and return out [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
    '''
    Llist = []
    alist = []
    blist = []
    if type(threetuple[0]) == type([]):
        for curlist in threetuple:
            Llist.append(curlist[0])
            alist.append(curlist[1])
            blist.append(curlist[2])
        return [Llist, alist, blist]
    else:
        return threetuple


def addpoints(listofpoints, axes, color='r', marker=',', s=20):
    '''
    given a list of points of form [[x1,y1,z1],...,[xn,yn,zn]] and axes,
    it will add the points to the axes.
    '''
    lablists = threeTupleToThreeLists(listofpoints)
    x, y, z = lablists[0], lablists[1], lablists[2]
    axes.scatter(x, y, z, color=color, marker=marker, s=s)


def RgbToXYZ(R, G, B):
    '''TEST THIS FUNCTION'''
    fractionR = R / 255.0
    fractionG = G / 255.0
    fractionB = B / 255.0

    if (fractionR > 0.04045):
        fractionR = ((fractionR + 0.055) / 1.055)**2.4
    else:
        fractionR = fractionR / 12.92
    if (fractionG > 0.04045):
        fractionG = ((fractionG + 0.055) / 1.055)**2.4
    else:
        fractionG = fractionG / 12.92
    if (fractionB > 0.04045):
        fractionB = ((fractionB + 0.055) / 1.055)**2.4
    else:
        fractionB = fractionB / 12.92

    fractionR = fractionR * 100
    fractionG = fractionG * 100
    fractionB = fractionB * 100

    X = fractionR * 0.4124 + fractionG * 0.3576 + fractionB * .1805
    Y = fractionR * 0.2126 + fractionG * 0.7152 + fractionB * .0722
    Z = fractionR * 0.0193 + fractionG * 0.1192 + fractionB * .9505

    return {"X": X, "Y": Y, "Z": Z}


def XyzToHunterLab(X, Y, Z):
    '''TEST THIS FUNCTION'''
    L = 10 * sqrt(Y)
    A = 17.5 * (((1.02 * X) - Y) / sqrt(Y))
    B = 7 * ((Y - (0.847 * Z)) / sqrt(Y))

    return {"L": L, "A": A, "B": B}


def RGBtoHunterLab(r, g, b):
    '''
    TEST THIS
    '''
    if (r == 0) and (g == 0) and (b == 0):
        return {"L": 0, "A": 0, "B": 0}
    else:
        xyz = RgbToXYZ(r, g, b)
        HLab = XyzToHunterLab(xyz["X"], xyz["Y"], xyz["Z"])
        return HLab


def HunterLabToXYZ(L, a, b):
    '''
    TESTED
    '''
    tempY = L / 10.0
    tempX = (a / 17.5) * (L / 10.0)
    tempZ = (b / 7.0) * (L / 10.0)

    Y = tempY ** 2
    X = (tempX + Y) / 1.02
    Z = (-1) * (tempZ - Y) / 0.847

    return X, Y, Z


def XYZToRGB(X, Y, Z):
    '''
    TESTED
    '''
    tempX = X / 100.0
    tempY = Y / 100.0
    tempZ = Z / 100.0

    tempR = (tempX * 3.2406) + (tempY * -1.5372) + (tempZ * -0.4986)
    tempG = (tempX * -0.9689) + (tempY * 1.8758) + (tempZ * 0.0415)
    tempB = (tempX * 0.0557) + (tempY * -0.2040) + (tempZ * 1.0570)

    if (tempR > 0.0031308):
        tempR = 1.055 * (tempR ** (1 / 2.4)) - 0.055
    else:
        tempR = 12.92 * tempR
    if (tempG > 0.0031308):
        tempG = 1.055 * (tempG ** (1 / 2.4)) - 0.055
    else:
        tempG = 12.92 * tempG
    if (tempB > 0.0031308):
        tempB = 1.055 * (tempB ** (1 / 2.4)) - 0.055
    else:
        tempB = 12.92 * tempB

    R = tempR * 255
    G = tempG * 255
    B = tempB * 255

    return R, G, B


def HunterLabToRGB(L, a, b, normalized=False):
    '''
    TESTED
    '''
    x, y, z = HunterLabToXYZ(L, a, b)
    R, G, B = XYZToRGB(x, y, z)

    if normalized == True:
        R = R / 255.0
        G = G / 255.0
        B = B / 255.0

    return R, G, B


def meanstdv(inputList):
    try:
        std = []
        Listlen = float(len(inputList))
        mean = (sum(inputList) / Listlen)
        SDsum = 0
        for value in inputList:
            SDsum += pow((value - mean), 2)
        stddev = sqrt(SDsum / Listlen)
        return [float(mean), float(stddev)]
    except Exception, e:
        print str(e)
        print " In Kernel.meanstdv ; does not compute"
        return "na", "na"


def clusterdistance(cluster1, cluster2):
    d1 = cluster1[0] - cluster2[0]
    d2 = cluster1[1] - cluster2[1]
    d3 = cluster1[2] - cluster2[2]
    sum = 0
    for distdiff in [d1, d2, d3]:
        distdiff = distdiff**2
        sum += distdiff
    return sqrt(sum)
