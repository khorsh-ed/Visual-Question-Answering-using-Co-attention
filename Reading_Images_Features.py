# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:36:47 2018

@author: Lenovo-PC
"""
import pandas as pd
import numpy as np

# Import The Dataset
def LoadFeatures(FilePath,FeaturesPath):
    with open(FilePath) as f:
        l = f.readlines()
    l = [x.strip() for x in l] 
    
    tmp = ['000000000000',
           '00000000000',
           '0000000000',
           '000000000',
           '00000000',
           '0000000',
           '000000',
           '00000',
           '0000',
           '000',
           '00',
           '0',
           ''
           ]
    DataSet_Features = np.full((len(l),512,49),0.0)
    for j in range(len(l)):
        dataset = pd.read_csv(FeaturesPath + tmp[len(str(l[j]))] + str(l[j]) + '.csv' , header = None)
        DataSet_Features[j,:,:] = dataset.iloc[:,:].values
    return DataSet_Features