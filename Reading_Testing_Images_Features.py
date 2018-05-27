# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:16:07 2018

@author: AbdullahMahmoud
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:36:47 2018

@author: Lenovo-PC
"""
import pandas as pd
import numpy as np

# Import The Dataset
def Testfun():
    with open('E:\\Prototype Dataset\\Testing\\img_ids.txt') as f:
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
        dataset = pd.read_csv('E:\\Prototype Dataset\\Testing Images Features\\COCO_val2014_' + tmp[len(str(l[j]))] + str(l[j]) + '.csv' , header = None)
        x = dataset.iloc[:, 0:3].values
        for i in range(len(x[:,0])):
            DataSet_Features[j,np.int32(x[i,1]),np.int32(x[i,2])] = x[i,0]
    return DataSet_Features