#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:59:55 2020

@author: siat
"""
import cellpose
from cellpose import models, plot
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
import pathlib, os, cv2
import skimage.io as io
from tqdm import tqdm

def findindex(pre_anchor, channel, pre_label):
    for item in pre_anchor:
        if item[channel] == pre_label:
            return item
    return -1
def maskinfer(img, channels, nuclei=False):
    
    cyto_model = models.Cellpose(gpu=True)
    nuclei_model = models.Cellpose(gpu=True, model_type='nuclei')
    
    ##### merged image #####
    if nuclei:
        masks, flows, _, _  = nuclei_model.eval(img, channels=channels, diameter=None)
    else:
        masks, flows, _, _  = cyto_model.eval(img, channels=channels, diameter=None)
    
    
    fig = plt.figure(figsize=(8,3), dpi=600)
    plot.show_segmentation(fig, img, masks, flows[0], channels=[[2, 1]], file_name='merged_estimatediam')

    return masks, flows
    ##### merged image(gray) #####
    # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # gray = np.sum(img, axis=2)
    # plt.imshow(gray, cmap='gray')
    
    
    ##### Chromatin #####
    '''
    chromatin: red crfp(channel 0)
    '''
    
    # chromatin = img[:, :, 0]
    # plt.imshow(chromatin, cmap='gray')
    # masks, flows, _, _  = nuclei_model.eval(chromatin, channels=[[0, 0]], diameter=60)
    # fig = plt.figure(figsize=(8,3), dpi=600)
    # plot.show_segmentation(fig, chromatin, masks, flows[0], channels=[[0, 0]], file_name='chromatin')
    
    ##### microtubles #####
    '''
    microtubes: green cgfp(channel 1)
    '''
    # microtubes = img[:, :, 1]
    # plt.imshow(microtubes, cmap='gray')
    # masks, flows, _, _  = cyto_model.eval(microtubes, channels=[[0, 0]], diameter=120)
    # fig = plt.figure(figsize=(8,3), dpi=600)
    # plot.show_segmentation(fig, microtubes, masks, flows[0], channels=[[0, 0]], file_name='microtubes')

def tailimgs(data, img, centx, centy, clahe):
    cropped_img = img[int(centx-32):int(centx+32), int(centy-32):int(centy+32), :].copy()
    cropped_img[:, :, 0] = clahe.apply(cropped_img[:, :, 0])
    cropped_img[:, :, 1] = clahe.apply(cropped_img[:, :, 1])
    # io.imsave(f'/home/siat/projects/meterial/{i}.png', cropped_img)
    cropped_img = mx.nd.expand_dims(mx.nd.array(cropped_img).transpose((2,0,1))/255.0, axis=0)
    data = mx.nd.Concat(data, cropped_img, dim=0)
    return data

def getcls(net, img, centx, centy, clahe, gpu=True, vector=False):
    cropped_img = img[int(centx-32):int(centx+32), int(centy-32):int(centy+32), :].copy()
    cropped_img[:, :, 0] = clahe.apply(cropped_img[:, :, 0])
    cropped_img[:, :, 1] = clahe.apply(cropped_img[:, :, 1])
    cropped_img = mx.nd.expand_dims(mx.nd.array(cropped_img).transpose((2,0,1))/255.0, axis=0)
    if gpu:
        cropped_img = cropped_img.copyto(mx.gpu())
    if vector:
        return net(cropped_img).asnumpy()
    else:
        return int(mx.nd.argmax(net(cropped_img), axis=1).asnumpy())
    
def getcenter(i, mask):
    xcord, ycord = np.where(mask == i)
    top = min(xcord)
    button = max(xcord)
    left = min(ycord)
    right = max(ycord)
    return int(top + (button - top)/2), int(left + (right - left)/2)

def maxioulabel(pre_mask, mask, label, threshold=0.3):
    iou = []
    bias = False
    intersect = (mask == label) * pre_mask
    ilist = np.unique(intersect)
    for i in ilist:
        if i == 0:
            bias = True
            continue
        else:
            iou.append(np.sum(intersect == i)/np.sum(pre_mask == i)) 
    if len(iou) > 0:
        if np.max(iou) > threshold:
            pre_label = ilist[np.argmax(iou)+1] if bias else ilist[np.argmax(iou)]
            return pre_label
        else: return -1
    else:
        return -1
    
def tracking(pre_mask, pre_nuc_mask, mask, nuc_mask, pre_anchor, anchor, label=None):
    copyanchor = anchor.copy()
    count = 0
    for item in copyanchor:
        cyto_label = item[4]
        if cyto_label == 0:
            # using nuclei segmentation compute iou
            nuc_label = item[3]
            pre_label = maxioulabel(pre_nuc_mask, nuc_mask, nuc_label, threshold=0.2)
            if label is not None:
                if pre_label in label:
                    item[5] = findindex(pre_anchor, 3, pre_label)[3]
                    count += 1
                    continue
                anchor.pop(count)
            else:
                item[5] = findindex(pre_anchor, 3, pre_label)[3]
        else:
            pre_cyto_label = maxioulabel(pre_mask, mask, cyto_label)
            pre_item = findindex(pre_anchor, 4, pre_cyto_label)
            if label is not None:
                if pre_item == -1:
                    anchor.pop(count)
                    continue
                elif pre_item[3] in label:
                    item[5] = pre_item[3]
                    count += 1
                    continue
                anchor.pop(count)
            else:
                item[5] = pre_item[3]
    return anchor

def distance():


    pass

def nuc_cyto(cover):
    model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')
    
    ### estimate nuclei diam; load nuclei model
    model_type = 'nuclei'
    diam_mean = 17
    pretrained_model = [os.fspath(model_dir.joinpath('%s_%d'%(model_type,j))) for j in range(4)]
    pretrained_size = os.fspath(model_dir.joinpath('size_%s_0.npy'%(model_type)))    
    nuc_model = models.CellposeModel(device=mx.gpu(), 
                                        pretrained_model=pretrained_model)
    sz_model = models.SizeModel(device=mx.gpu(),
                                pretrained_size=pretrained_size,
                                cp_model=nuc_model)
    diam, _ = sz_model.eval(cover[:, :, 0], channels=[[0, 0]])
    nuc_rescale = diam_mean / diam    
    
    ### estimate cyto diam; load nuclei model
    model_type = 'cyto'
    diam_mean = 30
    pretrained_model = [os.fspath(model_dir.joinpath('%s_%d'%(model_type,j))) for j in range(4)]
    pretrained_size = os.fspath(model_dir.joinpath('size_%s_0.npy'%(model_type)))    
    cyto_model = models.CellposeModel(device=mx.gpu(), 
                                      pretrained_model=pretrained_model)
    sz_model = models.SizeModel(device=mx.gpu(),
                                pretrained_size=pretrained_size,
                                cp_model=cyto_model)
    diam, _ = sz_model.eval(cover, channels=[[2, 1]])
    cyto_rescale = diam_mean / diam       
    return nuc_model, nuc_rescale, cyto_model, cyto_rescale


def seqinfer(imgs):
    f = open('tracking.txt', 'w')
    cover = imgs[0]
    h, w = cover.shape[0:2]
    
    clahe=cv2.createCLAHE(10, (8, 8))
    nuc_model, nuc_rescale, cyto_model, cyto_rescale = nuc_cyto(cover)
    pre_nuc_mask, _, _ = nuc_model.eval(cover[:, :, 0], rescale=nuc_rescale, channels=[[0, 0]],do_3D=False, net_avg=True)
    pre_mask, _, _ = cyto_model.eval(cover, rescale=cyto_rescale, channels=[[2, 1]],do_3D=False, net_avg=True)
    
    
    # mask propagation
    # get init position and label
    base = []
    anchor = []
    from clsmodel import loadmodel
    net = loadmodel()
    pre = -1
    frame = 0
    for i in np.unique(pre_nuc_mask):
        centx, centy = getcenter(i, pre_nuc_mask)
        if centx-32 < 1 or centx+32 > h or centy-32 < 1 or centy+32 > w:
            continue
        else:
            nuc_label = i
            cyto_label = pre_mask[centx, centy]
            item = [centx, centy, frame, nuc_label, cyto_label, pre, getcls(net, cover, centx, centy, clahe, vector=True)]
        anchor.append(item)
    base.append(anchor)
    
    
    #propagation
    for idx, img in enumerate(imgs[0:]):
        anchor = []
        frame = idx+1
        nuc_mask, _, _ = nuc_model.eval(img[:, :, 0], rescale=nuc_rescale, channels=[[0, 0]], do_3D=False, net_avg=True)
        mask, _, _ = cyto_model.eval(img, rescale=cyto_rescale, channels=[[2, 1]],do_3D=False, net_avg=True)
        for i in np.unique(nuc_mask):
            centx, centy = getcenter(i, nuc_mask)
            if centx-32 < 1 or centx+32 > h or centy-32 < 1 or centy+32 > w:
                continue
            else:
                nuc_label = i
                cyto_label = mask[centx, centy]
                item = [centx, centy, frame, nuc_label, cyto_label, pre, getcls(net, img, centx, centy, clahe, vector=True)]
            anchor.append(item)
        base.append(tracking(pre_mask, pre_nuc_mask, mask, nuc_mask, base[-1], anchor))
        
        pre_mask = mask
        pre_nuc_mask = nuc_mask
    
    f.write(str(base))
    f.close()


def seqinferwithlabel(imgs, label):
    from plot import seqplot
    cover = imgs[0]
    h, w = cover.shape[0:2]
    
    clahe=cv2.createCLAHE(10, (8, 8))
    nuc_model, nuc_rescale, cyto_model, cyto_rescale = nuc_cyto(cover)
    pre_nuc_mask, _, _ = nuc_model.eval(cover[:, :, 0], rescale=nuc_rescale, channels=[[0, 0]],do_3D=False, net_avg=True)
    pre_mask, _, _ = cyto_model.eval(cover, rescale=cyto_rescale, channels=[[2, 1]],do_3D=False, net_avg=True)
    
    
    base = []
    anchor = []
    from clsmodel import loadmodel
    net = loadmodel()
    pre = -1
    frame = 0
    i = np.unique(pre_nuc_mask)[label]
    centx, centy = getcenter(i, pre_nuc_mask)
    if centx-32 < 1 or centx+32 > h or centy-32 < 1 or centy+32 > w:
        print('segerror')
    else:
        nuc_label = i
        cyto_label = pre_mask[centx, centy]
        item = [centx, centy, frame, nuc_label, cyto_label, pre, getcls(net, cover, centx, centy, clahe, vector=False)]
    anchor.append(item)
    base.append(anchor)    
    for idx, img in enumerate(imgs[0:]):
        anchor = []
        frame = idx+1
        nuc_mask, _, _ = nuc_model.eval(img[:, :, 0], rescale=nuc_rescale, channels=[[0, 0]], do_3D=False, net_avg=True)
        mask, _, _ = cyto_model.eval(img, rescale=cyto_rescale, channels=[[2, 1]],do_3D=False, net_avg=True)
        for i in np.unique(nuc_mask):
            centx, centy = getcenter(i, nuc_mask)
            if centx-32 < 1 or centx+32 > h or centy-32 < 1 or centy+32 > w:
                continue
            else:
                nuc_label = i
                cyto_label = mask[centx, centy]
                item = [centx, centy, frame, nuc_label, cyto_label, pre, getcls(net, img, centx, centy, clahe, vector=False)]
            anchor.append(item)
        l = [base[-1][j][3] for j in range(len(base[-1]))]
        anchor = tracking(pre_mask, pre_nuc_mask, mask, nuc_mask, base[-1], anchor, l)
        base.append(anchor)
        pre_mask = mask
        pre_nuc_mask = nuc_mask
        seqplot(img, nuc_mask, anchor)







    
def plotflow(flows):
    fv = np.expand_dims(flows[1][0, :, :], axis=-1)/5*180
    fh = np.expand_dims(flows[1][1, :, :], axis=-1)/5*180
    
    canvas = np.ones_like(fv).repeat(3, axis=-1)*255
    canvas[fv.squeeze()>0, 1:] = 255 - np.expand_dims(fv[fv>0], axis=-1)
    canvas[fv.squeeze()<0, :2] = 255 + np.expand_dims(fv[fv<0], axis=-1)
    
    io.imsave('flowv.png', canvas)

    canvas = np.ones_like(fv).repeat(3, axis=-1)*255    
    canvas[fh.squeeze()>0, 1:] = 255 - np.expand_dims(fh[fh>0], axis=-1)
    canvas[fh.squeeze()<0, :2] = 255 + np.expand_dims(fh[fh<0], axis=-1)
    
    io.imsave('flowh.png', canvas)    
    
    
def load_data(url):
    imgs = []
    fnames = []
    for file in os.listdir(url):
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
# url = '/home/siat/projects/dataset/cellcog/H2b_aTub_MD20x_exp911/0040'  
# imgs, imgnames = load_data(url)
# imgs = np.array(imgs)
# argsort = np.argsort(imgnames)
# imgs = imgs[argsort]
# seqinferwithlabel(imgs, 50)  
  
    
    
    
    
    
    
    
    
    
    
    
    