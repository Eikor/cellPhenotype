#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:16:39 2020

@author: siat
"""

import mxnet as mx
from mxnet import nd, init, gluon, autograd
from mxnet.gluon import nn
import time
import numpy as np
# def mxnetclassifier():

    
def build():
    net = nn.Sequential()
    net.add(
        nn.Conv2D(16, 3, strides=1, padding=(1, 1), activation='relu', ),
        nn.BatchNorm(),
        nn.MaxPool2D(pool_size = 2, strides=2),
        nn.Conv2D(32, 3, strides=1, padding=(1, 1), activation='relu'),
        nn.BatchNorm(),
        nn.MaxPool2D(pool_size = 2, strides=2),
        nn.Conv2D(64, 3, strides=1, padding=(1, 1), activation='relu'),
        nn.BatchNorm(),
        nn.MaxPool2D(pool_size = 2, strides=2),      
        nn.AvgPool2D(pool_size = 2),
        nn.Conv2D(64, 3, strides=1, padding=(1, 1), activation='relu'),
        nn.BatchNorm(),
        nn.Conv2D(64, 3, strides=1, padding=(1, 1), activation='relu'),
        nn.BatchNorm(),
        nn.Conv2D(64, 3, strides=1, padding=(1, 1), activation='relu'),
        nn.BatchNorm(),        
        nn.AvgPool2D(pool_size = 2),
        nn.Dense(8)
            )
    
    return net
    

def buildandtrainnet():
    initializer = mx.initializer.Xavier()
    net = build()
    ctx = mx.gpu()
    net.initialize(initializer, ctx=ctx)
    net.summary(mx.nd.random.uniform(shape=(1, 3, 64, 64), ctx=ctx))

    batch_size = 64
    def transform(data, label):
        data = mx.image.imresize(data, 64, 64)
        return data.astype(np.float32).transpose((2,0,1))/255.0, np.float32(label)
    
    datasetpath = '/home/siat/projects/dataset/cellcog/H2b_aTub_MD20x_exp911/classifier/mc64'
    dataset1 = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(datasetpath, transform=transform), 
                                     batch_size=batch_size, shuffle=True, num_workers=8, last_batch='discard')
    dataset2 = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(datasetpath, transform=transform), 
                                     batch_size=batch_size, shuffle=True, num_workers=8, last_batch='discard')
    
    sgd_optimizer = mx.optimizer.SGD(learning_rate=0.001)
    trainer = gluon.Trainer(net.collect_params(), optimizer=sgd_optimizer)
    
    # start with epoch 1 for easier learning rate calculation
    metric = mx.metric.Accuracy()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    

    epochs = 100

    for epoch in range(1, epochs + 1):
        for (data1, data2) in zip(dataset1, dataset2):
            alpha = nd.array(np.random.beta(0.5, 0.5, (batch_size, 1, 1, 1)))
            data1, label1 = data1
            data2, label2 = data2
            data = alpha * data1 + (1-alpha) * data2
            alpha = alpha.reshape(batch_size, 1)
            label = alpha * nd.eye(8)[label1] + (1-alpha) * nd.eye(8)[label2]
            # get the images and labels
            data = gluon.utils.split_and_load(data, ctx_list=[ctx], batch_axis=0)
            label = gluon.utils.split_and_load(label, ctx_list=[ctx], batch_axis=0)
            with autograd.record():
                outputs = net(data[0]).softmax()
                loss = label[0]*outputs.log() + (1-label[0])*(1-outputs).log()
                loss = -nd.sum(loss)
 
            mx.autograd.backward(loss)
            trainer.step(batch_size)

        print(f"""Epoch[{epoch}] train loss = {loss.mean().asscalar()}""")
        
        val_loss = 0
        for data, label in dataset1:
            data = gluon.utils.split_and_load(data, ctx_list=[ctx], batch_axis=0)
            label = gluon.utils.split_and_load(label, ctx_list=[ctx], batch_axis=0)
            outputs = [net(data[0])]
            val_loss += softmax_cross_entropy_loss(outputs[0], label[0]).mean()/24
            metric.update(label, outputs)
        print(f"""Epoch[{epoch}] val loss = {val_loss.asscalar()}  acc: {metric.get()[1]}""")
    
    net.save_parameters('clsnet.params')
    

def loadmodel():    
    net = build()
    ctx = mx.gpu()
    net.load_parameters('clsnet.params', ctx=ctx)
    return net
