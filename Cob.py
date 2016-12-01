'''
Created on Sep 14, 2016

@author: Caleb Hulbert
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Kernel


class Cob(object):
    '''
    holds a list of Kernels, with associated statistics

    -name
    -kernellist         list of kernels
    -numkernels         int
    -mode               pixels or kernels
    -clusters           list of clusers of form [size, mean of point
    -mean
    '''

    def __init__(self, kernellist, name, clustertype="kmeans", pixelcluster=True, stats=False):
        self.name = name
        self.clustertype = clustertype
        self.kernellist = kernellist
        self.numkernels = len(self.kernellist)
        self.type = 'pixels'
        self.mean = [0, 0, 0]
        if pixelcluster == False:
            self.type = 'kernels'
            self.clusters = []
            self.kernelcenters = []
            self.setkernelcenters()
        if self.clustertype == "kmeans":
            self.clusters = []
            self.kmeans(pixelcluster=pixelcluster)
            self.mean = [0, 0, 0]
        elif self.clustertype == "dbscan":
            self.clusters = []
            self.dbscan()
        if stats == True:
            self.setstats()

    def setkernelcenters(self):
        '''
        create a list of all the centers calculated for the kernels this Cob contains
        '''
        for kernel in self.kernellist:
            for cluster in kernel.clusters:
                self.kernelcenters.append(cluster[1])

    def kmeans(self, pixelcluster=True):
        '''
        calculates kmeans centers for the cob from the centers of the kernels it contains
        '''
        if pixelcluster == True:
            pixellist = []
            for kernel in self.kernellist:
                pixellist.extend(kernel.pixellist)
            kmeans = Kernel.KMeans(n_clusters=2).fit(pixellist)
        else:
            kmeans = Kernel.KMeans(n_clusters=2).fit(self.kernelcenters)
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

    def setstats(self):
        '''
        calculates the mean in [L,a,b] form and set the the Cob.mean property
        '''
        L = 0
        a = 0
        b = 0
        numkernels = 0
        for kernel in self.kernellist:
            L += kernel.mean[0]
            a += kernel.mean[1]
            b += kernel.mean[2]
            numkernels += 1
        self.mean[0] = L / float(numkernels)
        self.mean[1] = a / float(numkernels)
        self.mean[2] = b / float(numkernels)

    def dbscan(self, eps=.5, plot=False):
        pixellist = []
        for kernel in self.kernellist:
            pixellist.extend(kernel.pixellist)
        X = Kernel.np.array(pixellist)
        numberofpixels = len(pixellist)
        density = eps
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
            size = len(llist)
            if size > numberofpixels / 100:
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

    def showscatterplot(self):
        '''
        creates a 3d scatter plot whose points are either pixels from all the kernels in this cob,
        or centers from the kmean clusters of each kernel in this cob.
        '''
        if self.type == 'pixels':
            pixels = []
            for kernel in self.kernellist:
                pixels.extend(kernel.pixellist)
            lablists = Kernel.threeTupleToThreeLists(pixels)
        else:
            lablists = Kernel.threeTupleToThreeLists(self.kernelcenters)
        plot = plt.figure()
        plt.close(1)
        del plot
        plot = plt.figure()
        axes = plot.add_subplot(111, projection='3d')
        llist = lablists[0]
        alist = lablists[1]
        blist = lablists[2]
        for l, a, b in zip(llist, alist, blist):
            R, G, B = Kernel.HunterLabToRGB(l, a, b, normalized=True)
            axes.scatter(l, a, b, color=[R, G, B], marker="o")
        axes.set_xlabel('L')
        axes.set_ylabel('a')
        axes.set_zlabel('b')
        for cluster in self.clusters:
            Kernel.addpoints(cluster[1], axes, marker="o", color="g")
        plt.ion()
        plt.show()
        return axes
