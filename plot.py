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
import cv2 

def plotoutline(img, masks, pred, anchor):
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
    # fig = plt.figure(dpi=600)
    # plt.imshow(imgout)

def plotmask(img, masks, pred, anchor, saveurl):
    canvas = np.zeros_like(img)
    for mask_id, label in zip(anchor, pred):
        mask = masks == mask_id
        l = label.asscalar()
        if l == 0: #apo
            canvas[mask] = np.array([255, 255, 255])
        elif l == 2: #inter #Lime
            canvas[mask] = np.array([10, 255, 130])
        elif l == 5: #pro #Cyan
            canvas[mask] = np.array([0, 255, 250])
        elif l == 6: #prometa 
            canvas[mask] = np.array([0, 190, 255])
        elif l == 4: #meta #blue
            canvas[mask] = np.array([175, 131, 255])
        elif l == 1: #earlyana #Magenta
            canvas[mask] = np.array([238, 149, 182])
        elif l == 3: #lateana #
            canvas[mask] = np.array([233, 168, 65])
        elif l == 7: #telo
            canvas[mask] = np.array([218, 247, 102])
    
    nuclei = img[:, :, 0][:, :, None]
    cell = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, None]
    # clahe=cv2.createCLAHE(40, (16, 16))
    # high_res = clahe.apply(nuclei)
    weights = nuclei / 255
    canvas = canvas * weights # (cell/np.max(cell)) # multiply nuclei
    io.imsave(saveurl, canvas)
    
    
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