'''
Created 11/22/2016
@authort Caleb Hulbert
'''
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


class Kernel(object):
    '''
    Built from a list of pixels(3 tuples), of the form (r,g,b) and converted to pixels of the
    form (l,a,b) where l,a,b are floats which make up a single hunter Lab color.
    Also contains statistical measure of the pixel list.

     Attributes:
        -name               string
        -pixellist          list of [l,a,b] lists
        -numberofpixels     integer
        -cluster1           [pixelsPercluster, Center]
        -cluster2           [pixelsPercluster, Center]
        -mean               [l,a,b]
        -mode               [l,a,b]
        -sd                 [lsd,asd,bsd] (standard deviations)
    '''

    def __init__(self, pixellist, name, cluster=True):
        self.name = name
        self.pixellist = []
        self.numberofpixels = 0
        self.mode = [0, 0, 0]
        self.mean = [0, 0, 0]
        self.sd = [0, 0, 0]
        self.setpixels(pixellist)
        self.type = 'pixels'
        if cluster == True:
            self.type = "kernels"
            self.cluster1 = [0, [0, 0, 0]]
            self.cluster2 = [0, [0, 0, 0]]
            self.setclusters()
            self.mode = [0, 0, 0]
            self.mean = [0, 0, 0]
            self.sd = [0, 0, 0]
        else: 
            self.setstats()
            

    def setpixels(self, pixellist):
        '''
        converts pixels from rgb to hunter lab and sets numberofpixels
        '''
        for pixel in pixellist:
            hlab = RGBtoHunterLab(pixel[0], pixel[1], pixel[2])
            self.pixellist.append([hlab["L"], hlab["A"], hlab["B"]])
        self.numberofpixels = len(self.pixellist)

    def setstats(self):
        '''
        calculates the mean, mode, and sd of the l, a, and b values in the pixellist.
        '''
        frequencydict = {}
        llist, alist, blist = [], [], []
        for pixel in self.pixellist:
            llist.append(pixel[0])
            alist.append(pixel[1])
            blist.append(pixel[2])
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

    def setclusters(self, default='mode'):
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
            self.cluster1[1] = default
            self.cluster2[1] = default
        else:
            self.cluster1[0] = sizec1
            self.cluster1[1] = kmeans.cluster_centers_[0].tolist()
            self.cluster2[0] = sizec2
            self.cluster2[1] = kmeans.cluster_centers_[1].tolist()

    def dbscan(self, eps=1.5, plot=False):
        X = StandardScaler().fit_transform(self.pixellist)
        db = DBSCAN(eps=eps).fit(self.pixellist)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print 'Estimated number of clusters: %d' % n_clusters_
        if plot is True:
            graph = plt.figure()
            ax = graph.add_subplot(111, projection='3d')
            unique_labels = set(labels)
            colors = plt.cm.get_cmap('Spectral')(
                np.linspace(0, 1, len(unique_labels)))
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = 'k'
                class_member_mask = (labels == k)
                xy = X[class_member_mask & core_samples_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col)
                xy = X[class_member_mask & ~core_samples_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col, marker='.')
            ax.set_xlabel('L')
            ax.set_ylabel('a')
            ax.set_zlabel('b')
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.ion()
            plt.show()

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
        if self.cluster1[1] != self.cluster2[1]:
            addpoints([self.cluster1[1], self.cluster2[1]], ax, marker='o')
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


def addpoints(listofpoints, axes, color='r', marker=','):
    '''
    given a list of points of form [[x1,y1,z1],...,[xn,yn,zn]] and axes,
    it will add the points to the axes.
    '''
    lablists = threeTupleToThreeLists(listofpoints)
    x, y, z = lablists[0], lablists[1], lablists[2]
    axes.scatter(x, y, z, color=color, marker=marker)


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
    except:
        print "does not compute"
        return "na", "na"
