'''
Created on Nov 25, 2016

@author: Caleb Hulbert
'''
import os
import csv
import tkFileDialog
from Tkinter import Tk
from sklearn.cluster import KMeans

import Cob
import Kernel


class Repline(object):
    '''
    contains a list of Cobs, along with some calculated statistics.

    -coblist           list of Cobs
    -directory      str(directory that the cobs where located in)
    -name           str(name of the rep line)
    -cluster1       [size, [L,a,b]]
    -cluster2       [size, [L,a,b]]
    -mean           [Lmean,amean,bmean]
    '''

    def __init__(self, startInDirectory='', row='', kernelcluster=False):
        Tk().withdraw()
        self.directory = ''
        self.name = ''
        self.setdirectoryandrow(startInDirectory, row)
        self.coblist = []
        self.cobcenters = []
        self.createcobs(kernelcluster=kernelcluster)
        self.setcobcenters()
        self.cluster1 = [0, [0, 0, 0]]
        self.cluster2 = [0, [0, 0, 0]]
        self.setclusters()
        self.mean = [0, 0, 0]
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

    def createcobs(self, kernelcluster=False):
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
                                kernellist.append(Kernel.Kernel(listofpixels, name=currentkernel, cluster=kernelcluster))
                                listofpixels=[]
                                currentkernel=int(line[1])
                                currentpixel=[int(line[2]), int(
                                    line[3]), int(line[4])]
                                listofpixels.append(currentpixel)
                            elif int(line[1]) == currentkernel:
                                currentpixel=[int(line[2]), int(
                                    line[3]), int(line[4])]
                                listofpixels.append(currentpixel)
                        except:
                            IndexError
                currentcob=Cob.Cob(kernellist, basename, pixelcluster = not kernelcluster)
                self.coblist.append(currentcob)

    def setcobcenters(self):
        '''
        create a list of all the centers calculated for the cobs this repline contains
        '''
        for cob in self.coblist:
            self.cobcenters.append(cob.cluster1[1])
            self.cobcenters.append(cob.cluster2[1])

    def setclusters(self):
        '''
        calculates kmeans centers for the repline from the centers of the cobs it contains
        '''
        if len(self.coblist) > 1:
            kmeans=KMeans(n_clusters = 2).fit(self.cobcenters)
            self.cluster1[1]=kmeans.cluster_centers_[0].tolist()
            self.cluster2[1]=kmeans.cluster_centers_[1].tolist()
            sizec1=0
            sizec2=0
            for label in kmeans.labels_:
                if label == 0:
                    sizec1 += 1
                elif label == 1:
                    sizec2 += 1
            self.cluster1[0]=sizec1
            self.cluster2[0]=sizec2
        else:
            self.cluster1[1]=self.coblist[0].cluster1[1]
            self.cluster2[1]=self.coblist[0].cluster2[1]

    def setstats(self):
        '''
        calculates and sets the [L,a,b] mean for this repline from the means of the cobs it contains.
        '''
        L=0
        a=0
        b=0
        numcobs=0
        for cob in self.coblist:
            L += cob.mean[0]
            a += cob.mean[1]
            b += cob.mean[2]
            numcobs += 1
        self.mean=[L / float(numcobs),
                     a / float(numcobs),
                     b / float(numcobs)]


def test():
    r = Repline(startInDirectory='..\src\TEST', row='A15LRH0_0012', kernelcluster=True)
    return r

if __name__ == '__main__':
    r = test()