'''
Created on Nov 25, 2016

@author: Caleb Hulbert
'''
import csv
import os
import tkFileDialog
from Tkinter import Tk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Cob
import Kernel


class Repline(object):
    '''
    contains a list of Cobs, along with some calculated statistics.
    clusterType options = ["kmeans", "dbscan"]

    -coblist           list of Cobs
    -directory      str(directory that the cobs where located in)
    -name           str(name of the rep line)
    -cluster        list of cluster of form [size, [mean of points]]
    -mean           [Lmean,amean,bmean]
    '''

    def __init__(self, startInDirectory='', row='', clustertype="kmeans", stats=False):
        Tk().withdraw()
        self.directory = ''
        self.name = ''
        self.clustertype = clustertype
        self.setdirectoryandrow(startInDirectory, row)
        self.coblist = []
        self.cobcenters = []
        self.createcobs(clustertype=clustertype, stats=stats)
        self.setcobcenters()
        self.clusters = []
        if self.clustertype == "kmeans":
            self.kmeans()
        elif self.clustertype == "dbscan":
            self.dbscan()
        self.mean = [0, 0, 0]
        if stats == True:
            self.setstats()

    def setdirectoryandrow(self, startindirectory='', row=''):
        '''
        ask user for row name and directory to find row files, unless they are given as an argument.
        '''
        if startindirectory == '':
            self.directory = str(tkFileDialog.askdirectory())
        else:
            self.directory = os.path.abspath(startindirectory)
        if row == '':
            filename = str(tkFileDialog.askopenfile(
                initialdir=self.directory).name)
            self.name = filename[len(self.directory) + 3:-8]
        else:
            self.name = row

    def createcobs(self, clustertype="kmeans", stats=False):
        '''
        looks at all the files in self.directory and finds and with the same base name as self.name.
        each file is turned into a cob object, and add to self.cobs
        '''
        for cobfile in os.listdir(self.directory):
            if '_' + self.name + "." in cobfile:
                filenamewithoutlastextension = os.path.splitext(cobfile)[0]
                basename = os.path.splitext(filenamewithoutlastextension)[0]
                kernellist = []
                with open(self.directory + "/" + cobfile) as csvfile:
                    csvreader = csv.reader(csvfile)
                    csvlist = list(csvreader)
                    listofpixels = []
                    currentkernel = 1
                    for line in csvlist:
                        try:
                            if line[0] == 'Image':
                                pass
                            elif int(line[1]) != currentkernel and line[4] != '':
                                kernellist.append(Kernel.Kernel(
                                    listofpixels, name=currentkernel, clustertype="dbscan", stats=stats))
                                listofpixels = []
                                currentkernel = int(line[1])
                                currentpixel = [int(line[2]), int(
                                    line[3]), int(line[4])]
                                listofpixels.append(currentpixel)
                            elif int(line[1]) == currentkernel:
                                currentpixel = [int(line[2]), int(
                                    line[3]), int(line[4])]
                                listofpixels.append(currentpixel)
                        except Exception, e:
                            print str(e)
                            IndexError
                currentcob = Cob.Cob(
                    kernellist, basename, pixelcluster=False, clustertype=clustertype, stats=stats)
                self.coblist.append(currentcob)

    def setcobcenters(self):
        '''
        create a list of all the centers calculated for the cobs this repline contains
        '''
        for cob in self.coblist:
            for cluster in cob.clusters:
                self.cobcenters.append(cluster[1])

    def kmeans(self):
        '''
        calculates kmeans centers for the repline from the centers of the cobs it contains
        '''
        if len(self.coblist) > 1:
            for cob in self.coblist:
                for cluster in cob.clusters:
                    self.cobcenters.append(cluster[1])
            kmeans = Kernel.KMeans(n_clusters=2).fit(self.cobcenters)
            meanc1 = kmeans.cluster_centers_[0].tolist()
            meanc2 = kmeans.cluster_centers_[1].tolist()
            sizec1 = 0
            sizec2 = 0
            for label in kmeans.labels_:
                if label == 0:
                    sizec1 += 1
                elif label == 1:
                    sizec2 += 1
            self.clusters.append([sizec1, meanc1])
            self.clusters.append([sizec2, meanc2])

        else:
            self.clusters = self.coblist[0].clusters

    def dbscan(self, eps=.5, plot=False):
        if len(self.coblist) > 1:
            pixlist = []
            totalnumkernels = 0
            for cob in self.coblist:
                for kernel in cob.kernellist:
                    pixlist.extend(kernel.pixellist)
                    totalnumkernels += 1
            X = Kernel.np.array(pixlist)
            density = eps
            numberofpixels = len(pixlist)
            db = Kernel.DBSCAN(eps=density).fit(X)
            core_samples_mask = Kernel.np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            unique_labels = set(labels)
            colors = []
            for k in unique_labels:
                class_member_mask = (labels == k)
                xyz = X[class_member_mask & core_samples_mask]
                llist, alist, blist = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                if len(llist) >= numberofpixels / 100:
                    lmean = llist.mean()
                    amean = alist.mean()
                    bmean = blist.mean()
                    self.clusters.append([size, [lmean, amean, bmean]])
                    r, g, b = Kernel.HunterLabToRGB(lmean, amean, bmean)
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
                    if len(xyz[:, 0]) >= numberofpixels / 100:
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
        elif len(self.coblist) == 1:
            self.clusters = self.coblist[0].clusters

    def setstats(self):
        '''
        calculates and sets the [L,a,b] mean for this repline from the means of the cobs it contains.
        '''
        L = 0
        a = 0
        b = 0
        numcobs = 0
        for cob in self.coblist:
            L += cob.mean[0]
            a += cob.mean[1]
            b += cob.mean[2]
            numcobs += 1
        self.mean = [L / float(numcobs),
                     a / float(numcobs),
                     b / float(numcobs)]

    def checkdistance(self):
        c1 = self.clusters[0]
        c2 = self.clusters[1]
        dist = Kernel.clusterdistance(c1[1], c2[1])
        if dist < 7.5:
            L = (c1[0] * c1[1][0] + c2[0] * c2[1][0]) / (c1[0] + c2[0])
            a = (c1[0] * c1[1][1] + c2[0] * c2[1][1]) / (c1[0] + c2[0])
            b = (c1[0] * c1[1][2] + c2[0] * c2[1][2]) / (c1[0] + c2[0])
            self.cluster = [[c1[0] + c2[0]], [L, a, b]]
            self.segmenting = True
        else:
            self.cluster = [0,[0,0,0]]
        return dist


def test(clustertype="dbscan", stats=False, rownum = 23):
    # r = Repline(startInDirectory='..\src\TEST',
    #             row='A15LRH0_0012', clustertype=clustertype, stats=stats)
    if rownum > 99:
        row = "A15LRH0_0" + str(rownum)
    elif rownum > 9:
        row = "A15LRH0_00" + str(rownum)
    elif rownum > -1:
        row = "A15LRH0_000" + str(rownum)
    from time import clock
    c1 = clock()
    r = Repline(startInDirectory='C:/Users/cmhul/Google Drive/College_/Corn_Color_Phenotyping/Hybrid_Phenotyping/Kernel CSVs',
                clustertype="kmeans", row=row)
    print clock() - c1
    r.checkdistance()
    return r

if __name__ == '__main__':
    # r = test("dbscan")
    r = test()
    pass
