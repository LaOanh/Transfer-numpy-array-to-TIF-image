
import numpy as np
from numpy import load, save, concatenate
import pandas as pd
import tkinter as tkinter
from tkinter import filedialog
from sklearn import preprocessing
from tkinter.filedialog import askopenfilename
# Loading the TOA_angle_AOT patch (3*3)
test_paths = filedialog.askopenfilenames(title='Choose  testing files', filetypes=[("NPY", ".npy")])
y_prediction = np.load(test_paths[00])


pred_B1 = y_prediction[:, 0]
pred_B2 = y_prediction[:, 1]
pred_B3 = y_prediction[:, 2]
pred_B4 = y_prediction[:, 3]
pred_B5 = y_prediction[:, 4]

## Reshape into image
pred_B1_reshape = np.reshape(pred_B1, (1, -1))
pred_B1_2d = np.reshape(pred_B1_reshape, (1330-3, 1549-3), order='C')
pred_B2_reshape = np.reshape(pred_B2, (1, -1))
pred_B2_2d = np.reshape(pred_B2_reshape, (1330-3, 1549-3), order='C')
pred_B3_reshape = np.reshape(pred_B3, (1, -1))
pred_B3_2d = np.reshape(pred_B3_reshape, (1330-3, 1549-3), order='C')
pred_B4_reshape = np.reshape(pred_B4, (1, -1))
pred_B4_2d = np.reshape(pred_B4_reshape, (1330-3, 1549-3), order='C')
pred_B5_reshape = np.reshape(pred_B5, (1, -1))
pred_B5_2d = np.reshape(pred_B5_reshape, (1330-3, 1549-3), order='C')

filelist2 = np.concatenate((pred_B1_2d, pred_B2_2d, pred_B3_2d, pred_B4_2d, pred_B5_2d), axis=0)
filelist_reshape2 = filelist2.reshape(5, pred_B1_2d.shape[0], pred_B1_2d.shape[1]).astype('float32')


## SAVING to TIF image
img_path = askopenfilename(title=u'Open image file', filetypes=[("TIF", ".tif")])
import rasterio as rio
with rio.open(img_path) as src:  # choose one image
    out_data = src.read()
    out_meta = src.meta
out_meta.update(count=len(filelist_reshape2))
out_meta['dtype'] = "float32"
out_meta['No Data'] = 0.0

Fname = tkinter.filedialog.asksaveasfilename(title=u'Save clip image file', filetypes=[("TIF", ".tif")])
with rio.open(Fname, "w", **out_meta) as dst:
    dst.write(filelist_reshape2)