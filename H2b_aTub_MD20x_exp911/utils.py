# -*- coding: utf-8 -*-
import skimage
import skimage.io as io
import os
import matplotlib.pyplot as plt
import numpy as np

url = './H2b_aTub_MD20x_exp911/classifier/samples'
labels = os.listdir(url)

# dataset = []
# for label in labels:
#     path = os.path.join(url, label)
#     images = []
#     for image in os.listdir(path):
#         images.append(os.path.join(path, image))
#     dataset.append(images)
    
# # count each label's resolution
# print(labels)
# color = ['fuchsia', 'lawngreen', 'orange', 'blueviolet', 'green', 'red', 'limegreen', 'yellow']

# fig = plt.figure(figsize=(8, 6), dpi=100)
# for idx, each_ in enumerate(dataset):
#     #each_ : list of images with same label
    
#     resolution = []
#     for u in each_:
#         resolution.append(io.imread(u).shape)
#     resolution = np.array(resolution)
#     plt.scatter(resolution[:, 0], resolution[:, 1], s=1, alpha=0.6, c=color[idx])

# fig_c, ax = plt.subplots(figsize=(8, 6), dpi=200)
# for idx, each_ in enumerate(dataset):
#     ax.bar(idx, len(each_), color=color[idx])
# ax.set_xticks(range(len(dataset)))
# ax.set_xticklabels(labels)

### data augmentation
"""
latenan  x2
pro      x2
meta     x2
earlyana x4
inter    x1
apo      x1
telo     x1.5
prometa  x2
"""
def dataaug(url, labels):
    newurl = './H2b_aTub_MD20x_exp911_classifier/merged'
    for label in labels:
        read_url = url + '/' + label + '/'
        save_url = os.path.join(newurl, label)
        if not os.path.exists(save_url):
            os.makedirs(save_url)
        
        if label in ['lateana', 'pro', 'meta', 'prometa']:   
            for fname in os.listdir(read_url):
                if fname.endswith('img.png'):
                    i = io.imread(os.path.join(read_url, fname))
                    io.imsave(os.path.join(save_url, fname), i)
                    io.imsave(os.path.join(save_url, fname +'lr_img.png'), i[:, ::-1])
                
        if label in ['earlyana']:
            for fname in os.listdir(read_url):
                if fname.endswith('img.png'):
                    i = io.imread(os.path.join(read_url, fname))
                    io.imsave(os.path.join(save_url, fname), i)
                    io.imsave(os.path.join(save_url, fname +'lr_img.png'), i[:, ::-1])
                    io.imsave(os.path.join(save_url, fname +'tb_img.png'), i[::-1, :])
                    io.imsave(os.path.join(save_url, fname +'tblr_img.png'), i[::-1, ::-1])
        
        if label in ['inter', 'apo']:
            for fname in os.listdir(read_url):
                if fname.endswith('img.png'):
                    i = io.imread(os.path.join(read_url, fname))
                    io.imsave(os.path.join(save_url, fname), i)
        
        if label in ['telo']:
            f = True
            for fname in os.listdir(read_url):
                if fname.endswith('img.png'):
                    i = io.imread(os.path.join(read_url, fname))
                    io.imsave(os.path.join(save_url, fname), i)
                    if f:
                        io.imsave(os.path.join(save_url, fname +'lr_img.png'), i[:, ::-1])
                        f = not f
                    else:
                        f = not f

            
def cropROI(url, labels, he=False):
    import cv2
    '''
    crop nuclear and cyto
    file name: P00**_T00***_X****_Y****__img.png
                trial_time_Xcord_Ycord 
    crop 64*64 windows on nuclear channel and cyto channel
    '''
    newurl = './H2b_aTub_MD20x_exp911/classifier/mc64'
    imgurl = './H2b_aTub_MD20x_exp911'
    if he:
        clahe=cv2.createCLAHE(10, (8, 8))
    for label in labels:
        read_url = os.path.join(url, label)
        save_url = os.path.join(newurl, label)
        if not os.path.exists(save_url):
            os.makedirs(save_url)
        for fname in os.listdir(read_url):
            name = fname.split('_')
            pid = name[0]
            t = name[1]
            X = int(name[2][1:])
            Y = int(name[3][1:])
            
            crfp_url = imgurl + '/' + pid[1:] + '/' + 'tubulin_' + pid + '_' + t +'_Crfp_Z1_S1.tif'
            cgfp_url = imgurl + '/' + pid[1:] + '/' + 'tubulin_' + pid + '_' + t +'_Cgfp_Z1_S1.tif'
            if he: 
                crfp = io.imread(crfp_url, -1)
                cgfp = io.imread(cgfp_url, -1)
            else:
                crfp = io.imread(crfp_url, -1)
                cgfp = io.imread(cgfp_url, -1)
            try:
                roi = np.stack([
                    clahe.apply(crfp[Y-32 : Y+32, X-32 : X+32]), 
                    clahe.apply(cgfp[Y-32 : Y+32, X-32 : X+32]), 
                    np.zeros((64, 64))
                    ], axis=-1)
            except :
                print(fname)
            if 'lr' in fname:
                if 'tb' in fname:
                    io.imsave(os.path.join(save_url, fname), roi[::-1, ::-1, :])
                else:
                    io.imsave(os.path.join(save_url, fname), roi[:, ::-1, :])
            elif 'tb' in fname:
                io.imsave(os.path.join(save_url, fname), roi[::-1, :, :])
            else:
                io.imsave(os.path.join(save_url, fname), roi)
            # print(roi.shape)
            # plt.figure(dpi=500)
            # plt.imshow(io.imread(img_url, -1), cmap='gray', zorder=0)
            # plt.scatter(X, Y, zorder=1, color='red', s=1)
            # plt.show()
            
            
dataaug(url, labels)
cropROI('./H2b_aTub_MD20x_exp911/classifier/merged', labels, he=True)

















              
