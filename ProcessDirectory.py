'''
Created on Nov, 29 2016

@author: Caleb Hulbert
'''
from Tkinter import Tk
import tkFileDialog
import os

import Repline


def start(startindirectory='', write=True, clustertype='kmeans', stats='False'):
    Tk().withdraw()
    if startindirectory == '':
        startindirectory = str(tkFileDialog.askdirectory())
    repnames = []
    for cobfile in os.listdir(startindirectory):
        if ".tif.csv" in cobfile and "1_" in cobfile:
            # Remove .csv
            repname = os.path.splitext(cobfile)[0]
            # Remove .tif
            repname = os.path.splitext(repname)[0]
            #remove leading number
            repname = repname[repname.find('_')+1:]
            repnames.append(repname)
    replines = []
    print "Constructing Replines"
    for namenum in xrange(len(repnames)):
        repline = Repline.Repline(
            startInDirectory=startindirectory, row=repnames[namenum], clustertype=clustertype, stats=stats)
        replines.append(repline)
        if namenum < len(repnames) - 1:
            print "(", namenum + 1, "/", len(repnames), ")", "Finished: ", repline.name, "Constructing ", repnames[namenum + 1], "..."
        else:
            print "(", namenum + 1, "/", len(repnames), ")", "Finished: ", repline.name
            print "Done!"
    return replines


if __name__ == '__main__':
    start()
