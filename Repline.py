'''
Created on Nov 25, 2016

@author: Caleb Hulbert
'''

from Tkinter import Tk
import tkFileDialog
import os
import csv

import Cob
import Kernel


class Repline(object):
    '''
    contains a list of Cobs, along with some calculated statistics.

    -cobs           list of Cobs
    -directory      str(directory that the cobs where located in)
    -name           str(name of the rep line)
    -cluster1       [size, [L,a,b]]
    -cluster2       [size, [L,a,b]]
    -mean           [Lmean,amean,bmean]
    '''

    def __init__(self, startInDirectory='', row=''):
        Tk.withdraw()
        self.directory = ''
        self.name = ''
        self.setdirectoryandrow(startInDirectory, row)
        self.createcobs()
        self.setclusters()
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

    def createcobs(self):
        '''
        looks at all the files in self.directory and finds and with the same base name as self.name.
        each file is turned into a cob object, and add to self.cobs
        '''
        cobs = []
        for cobfile in os.listdir(self.directory):
            if '_' + self.name + "." in cobfile:
                filenamewithoutlastextension = os.path.splitext(cobfile)[0]
                basename = os.path.splitext(filenamewithoutlastextension)[0]
                with open(self.directory + cobfile) as csvfile:
                    csvreader = csv.reader(csvfile)