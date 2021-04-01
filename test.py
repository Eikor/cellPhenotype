#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:00:22 2020

@author: siat
"""

import cellpose
from cellpose import models, plot
import skimage.io as io
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def clahe_data(url):
    ''' 
    For H2b_aTub_MD20x_exp911_0038 dataset
    file name: 
        tubulin_P0038_T00***_C*fp_Z1_S1.tif
    return:
        array of 2D images
    '''
    imgs = []
    fnames = []
    clahe=cv.createCLAHE(40, (16, 16))
    for file in os.listdir(url):
        if file.endswith('rfp_Z1_S1.tif'):
            imgurl = os.path.join(url, file)
            crfp = clahe.apply(io.imread(imgurl, -1))
            imgurl = imgurl.replace('Crfp', 'Cgfp')
            cgfp = clahe.apply(io.imread(imgurl, -1))
            
            color_img = np.zeros_like(np.expand_dims(cgfp, axis=2)).repeat(3, axis=2)
            color_img[:, :, 0] = crfp
            color_img[:, :, 1] = cgfp
            fnames.append(file)
            imgs.append(color_img)
    return imgs, fnames

def load_data(url):
    imgs = []
    fnames = []
    urls = os.listdir(url)
    urls.sort()
    for file in urls:
        if file.endswith('rfp_Z1_S1.tif'):
            imgurl = os.path.join(url, file)
            crfp = io.imread(imgurl, -1)
            imgurl = imgurl.replace('Crfp', 'Cgfp')
            cgfp = io.imread(imgurl, -1)
            
            color_img = np.zeros_like(np.expand_dims(cgfp, axis=2)).repeat(3, axis=2)
            color_img[:, :, 0] = crfp
            color_img[:, :, 1] = cgfp
            fnames.append(file)
            imgs.append(color_img)
    return imgs, fnames

def seq_analysis(imgs, imgnames):
    imgs = np.array(imgs)
    argsort = np.argsort(imgnames)
    imgs = imgs[argsort]
    from maskinfer import seqinfer
    seqinfer(imgs)

def rxrx(imgname):
    from maskinfer import plotflow
    cmap = plt.get_cmap('plasma')
    img = io.imread(imgname)
    masks, flows = maskinfer(img, [[1, 3]])
    plotflow(flows)
    canvas = np.zeros_like(masks)
    canvas = np.expand_dims(canvas, -1).repeat(3, axis=-1)
    color = np.arange(np.max(masks))/np.max(masks)
    np.random.shuffle(color)
    for i in np.unique(masks)[1:]:
        canvas[masks==i, :] = np.array(cmap(color[i-1])[:3])*255
    io.imsave('masks.png', canvas)

if __name__ == '__main__':
    from maskinfer import maskinfer
    import mxnet as mx
    from clsmodel import loadmodel
    from plot import plotmask
    
    url = './H2b_aTub_MD20x_exp911/0040'  
    imgs, imgnames = load_data(url) # load entire datasets

    index = 0 # image index
    nuclei = imgs[index][:, :, 0] 
    h, w = nuclei.shape[0:2]

    masks, flows = maskinfer(nuclei, [[0, 0]], nuclei=True) # using cellpose to segment cell image


    ctx = mx.gpu()
    net = loadmodel()

    clahe=cv.createCLAHE(10, (8, 8))
    labels = ['apo', 'earlyana', 'inter', 'lateana', 'meta', 'pro', 'prometa', 'telo']
    img = imgs[index].copy()
    data = mx.nd.zeros((1, 3, 64, 64))
    anchor = []
    for i in np.unique(masks):
        # crop each cell & send to classification modual
        mask = masks==i
        xcord, ycord = np.where(masks==i)
        top = min(xcord)
        button = max(xcord)
        left = min(ycord)
        right = max(ycord)
        centx = top + (button - top)/2
        centy = left + (right - left)/2
        if centx-32 < 1 or centx+32 > h or centy-32 < 1 or centy+32 > w:
            continue # exclude cells on border
        else:
            anchor.append(i)
            cropped_img = img[int(centx-32):int(centx+32), int(centy-32):int(centy+32), :].copy()
            cropped_img[:, :, 0] = clahe.apply(cropped_img[:, :, 0])
            cropped_img[:, :, 1] = clahe.apply(cropped_img[:, :, 1])
            # io.imsave(f'/home/siat/projects/meterial/{i}.png', cropped_img)
            cropped_img = mx.nd.expand_dims(mx.nd.array(cropped_img).transpose((2,0,1))/255.0, axis=0)
            data = mx.nd.Concat(data, cropped_img, dim=0)

    data = data.copyto(ctx)
    pred = mx.nd.argmax(net(data), axis=1).astype('int')
    pred = pred[1:]


    plotmask(img, masks, pred, anchor, f'out/{index}.png')
    pred = pred.asnumpy()
    count = []
    for l in range(len(labels)):
        count.append(np.count_nonzero(pred == l))

    plt.bar(range(len(labels)), count, tick_label = labels)
    plt.savefig(f'out/static_{index}.png')
    



