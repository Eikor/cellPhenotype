#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:42:01 2020

@author: siat
"""

from cellpose import utils
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def plot(img, masks, pred, anchor):
    outlines = utils.masks_to_outlines(masks)
    outX, outY = np.nonzero(outlines)
    imgout = img.copy()
    for x, y in zip(outX, outY):
        try:
            l = pred[anchor.index(masks[x, y])].asscalar()
            if l == 0: #apo
                imgout[x, y] = np.array([255, 255, 255])
            elif l == 2: #inter #Lime
                imgout[x, y] = np.array([0, 255, 0])
            elif l == 5: #pro #Cyan
                imgout[x, y] = np.array([0, 255, 255])
            elif l == 6: #prometa 
                imgout[x, y] = np.array([0, 100, 255])
            elif l == 4: #meta #blue
                imgout[x, y] = np.array([0, 0, 255])
            elif l == 1: #earlyana #Magenta
                imgout[x, y] = np.array([255, 0, 255])
            elif l == 3: #lateana #
                imgout[x, y] = np.array([255, 0, 0])
            elif l == 7: #telo
                imgout[x, y] = np.array([255, 255, 0])
            pass
        except:
            pass
    pass
    io.imwrite('seqout/est.png', imgout)
    fig = plt.figure(dpi=600)
    plt.imshow(imgout)
    
def seqplot(img, mask, anchor):
    targets = [[anchor[j][3], anchor[j][-1]] for j in range(len(anchor))]
    imgout = img.copy()
    for i, l in targets:
        canvas = mask == i
        outlines = utils.masks_to_outlines(canvas)
        outX, outY = np.nonzero(outlines)
        
        for x, y in zip(outX, outY):
            try:
                if l == 0: #apo
                    imgout[x, y] = np.array([255, 255, 255])
                elif l == 2: #inter #Lime
                    imgout[x, y] = np.array([0, 255, 0])
                elif l == 5: #pro #Cyan
                    imgout[x, y] = np.array([0, 255, 255])
                elif l == 6: #prometa 
                    imgout[x, y] = np.array([0, 100, 255])
                elif l == 4: #meta #blue
                    imgout[x, y] = np.array([0, 0, 255])
                elif l == 1: #earlyana #Magenta
                    imgout[x, y] = np.array([255, 0, 255])
                elif l == 3: #lateana #
                    imgout[x, y] = np.array([255, 0, 0])
                elif l == 7: #telo
                    imgout[x, y] = np.array([255, 255, 0])
                pass
            except:
                pass
        pass
    io.imsave(f'seqout/{str(anchor[0][2])}.png', imgout)