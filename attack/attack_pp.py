#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import cv2
import sys
import math
import numpy as np
import argparse
import functools
import time
import paddle
import paddle.fluid as fluid
from utils import *
import six
import pandas as pd


#实现linf约束 输入格式都是tensor 返回也是tensor [1,3,224,224]
def linf_img_tenosr(o,adv,epsilon=16.0/256):
    
    o_img=tensor2img(o)
    adv_img=tensor2img(adv)
    
    clip_max=np.clip(o_img*(1.0+epsilon),0,255)
    clip_min=np.clip(o_img*(1.0-epsilon),0,255)
    
    adv_img=np.clip(adv_img,clip_min,clip_max)
    
    adv_img=img2tensor(adv_img)
    
    return adv_img
"""
Explaining and Harnessing Adversarial Examples, I. Goodfellow et al., ICLR 2015
实现了FGSM 支持定向和非定向攻击的单步FGSM


input_layer:输入层
output_layer:输出层
step_size:攻击步长
adv_program：生成对抗样本的prog 
eval_program:预测用的prog
isTarget：是否定向攻击
target_label：定向攻击标签
epsilon:约束linf大小
o:原始数据
use_gpu：是否使用GPU

返回：
生成的对抗样本
"""
def FGSM(adv_program,eval_program,gradients,o,input_layer,output_layer,step_size=16.0/256,epsilon=16.0/256,isTarget=False,target_label=0,use_gpu=False):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:o })
    result = result[0][0]
   
    o_label = np.argsort(result)[::-1][:1][0]
    
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label=o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    #计算梯度
    g = exe.run(adv_program,
                     fetch_list=[gradients],
                     feed={ input_layer.name:o,'label': target_label  }
               )
    g = g[0][0]
    
    print(g.shape)
    
    if isTarget:
        adv=o-np.sign(g)*step_size
    else:
        adv=o+np.sign(g)*step_size
    
    #实施linf约束
    adv=linf_img_tenosr(o,adv,epsilon)
    
    return adv


"""
Towards deep learning models resistant to adversarial attacks, A. Madry, A. Makelov, L. Schmidt, D. Tsipras, 
and A. Vladu, ICLR 2018
实现了PGD 支持定向和非定向攻击的PGD


input_layer:输入层
output_layer:输出层
step_size:攻击步长
adv_program：生成对抗样本的prog 
eval_program:预测用的prog
isTarget：是否定向攻击
target_label：定向攻击标签
epsilon:约束linf大小
o:原始数据
use_gpu：是否使用GPU

返回：
生成的对抗样本
"""
def PGD(adv_program,eval_program,gradients,o,input_layer,output_layer,step_size=2.0/256,epsilon=16.0/256,iteration=10,isTarget=False,target_label=0,use_gpu=True):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:o })
    result = result[0][0]
   
    o_label = np.argsort(result)[::-1][:1][0]
    
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label=o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    adv=o.copy()
    
    for _ in range(iteration):
    
        #计算梯度
        g = exe.run(adv_program,
                         fetch_list=[gradients],
                         feed={ input_layer.name:adv,'label': target_label  }
                   )
        g = g[0][0]

        if isTarget:
            adv=adv-np.sign(g)*step_size
        else:
            adv=adv+np.sign(g)*step_size
    
    #实施linf约束
    adv=linf_img_tenosr(o,adv,epsilon)
    
    return adv


def PGD2(adv_program,eval_program,gradients,o,input_layer,output_layer,src_label,step_size=2.0/256,epsilon=16.0/256,iteration=20,isTarget=False,target_label=0,use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    o_label = src_label
    
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label=o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    adv=o.copy()
    
    for _ in range(iteration):
    
        #计算梯度
        g = exe.run(adv_program,
                         fetch_list=[gradients],
                         feed={ input_layer.name:adv,'label': target_label  }
                   )
        g = g[0][0]

        if isTarget:
            adv=adv-np.sign(g)*step_size
        else:
            adv=adv+np.sign(g)*step_size
    
    #实施linf约束
    adv=linf_img_tenosr(o,adv,epsilon)
    
    return adv

def PGD3(adv_program,eval_program,gradients,o,input_layer,output_layer,src_label,step_size=2.0/256,epsilon=16.0/256,iteration=20,isTarget=False,target_label=0,use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:o })
    result = result[0][0]
    #print(result)
    #print(np.argsort(result)[::-1])
    
    #print(np.argsort(result)[::-1][-1])
    o_label = np.argsort(result)[::-1][-15]#modif
    #print(o_label)
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label=o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    #print(target_label.shape)
    target_label=np.expand_dims(target_label, axis=0)
    #print(target_label.shape)
    #print(target_label)
    adv=o.copy()
    
    for _ in range(iteration):
    
        #计算梯度
        g = exe.run(adv_program,
                         fetch_list=[gradients],
                         feed={ input_layer.name:adv,'label': target_label  }
                   )
        g = g[0][0]
        print(g.shape)
        if isTarget:
            adv=adv-np.sign(g)*step_size
        else:
            adv=adv-np.sign(g)*step_size#modified
    
    #实施linf约束
    adv=linf_img_tenosr(o,adv,epsilon)
    
    return adv


def DEEP_FOOL(adv_program,eval_program,gradients,o,input_layer,output_layer,src_label,step_size=2.0/256,epsilon=16.0/256,iteration=20,isTarget=False,target_label=0,use_gpu=True):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:o })
    result = result[0][0]
   
    #o_label = np.argsort(result)[::-1][:1][0]
    o_label = src_label
    
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label=o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    #print(target_label)
    adv=o.copy()
    g = exe.run(adv_program,
                         fetch_list=[gradients],
                         feed={ input_layer.name:adv,'label': target_label  }
                   )
    g = g[0][0]


    labels = [i+1 for i in range(120)]
    pred_label = target_label[0][0]
    while pred_label == target_label[0][0]:
        print("begin")
        w = np.inf
        w_norm = np.inf
        pert = np.inf
        for k in labels:
            if k == target_label:
                continue
        #计算梯度
            k_label=np.array([k]).astype('int64')
            k_label=np.expand_dims(k_label, axis=0)
            g_k = exe.run(adv_program,
                         fetch_list=[gradients],
                         feed={ input_layer.name:adv,'label': k_label  }
                   )
            g_k = g_k[0][0]
            w_k = g_k - g
            #print("w_k is {}".format(w_k))
            f_k = result[k] -  result[target_label][0][0]
            #print("f_k is ",f_k)
            #print("result[k] is ",result[k])
            #print("result[target_label] is ",result[target_label])
            w_k_norm = np.linalg.norm(w_k.flatten()) 
            #print("w_k_norm is ",w_k_norm)
            pert_k = (np.abs(f_k) + 1e-8) / w_k_norm
            #print("pert_k is ",pert_k)
            if pert_k < pert:
                pert = pert_k
                w = w_k
                w_norm = w_k_norm
        r_i = w * pert / w_norm
        #print("r_i is {}".format(r_i))
        overshoot=10
        
        if isTarget:
            adv=adv-np.sign(g)*step_size
        else:
            adv=adv-(1 + overshoot)*r_i
    
    
    #实施linf约束
        adv=linf_img_tenosr(o,adv,epsilon)
        resul = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:adv })
        resul = resul[0][0]
        pred_label = np.argmax(resul)
    
    return adv

    
def ensem_mom_attack_threshold_8model(adv_program,eval_program,gradients,o,src_label2,src_label,out1,out2,out3,out4,out5,out6,out7,out8,label,iteration=20,use_gpu = True):
    #将小扰动调整为0
    origdict = {1: 0, 2: 32, 3: 43, 4: 54, 5: 65, 6: 76, 7: 87, 8: 98, 9: 109, 10: 1, 11: 12, 12: 23, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29, 18: 30, 19: 31, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 28: 41, 29: 42, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48, 35: 49, 36: 50, 37: 51, 38: 52, 39: 53, 40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 60, 46: 61, 47: 62, 48: 63, 49: 64, 50: 66, 51: 67, 52: 68, 53: 69, 54: 70, 55: 71, 56: 72, 57: 73, 58: 74, 59: 75, 60: 77, 61: 78, 62: 79, 63: 80, 64: 81, 65: 82, 66: 83, 67: 84, 68: 85, 69: 86, 70: 88, 71: 89, 72: 90, 73: 91, 74: 92, 75: 93, 76: 94, 77: 95, 78: 96, 79: 97, 80: 99, 81: 100, 82: 101, 83: 102, 84: 103, 85: 104, 86: 105, 87: 106, 88: 107, 89: 108, 90: 110, 91: 111, 92: 112, 93: 113, 94: 114, 95: 115, 96: 116, 97: 117, 98: 118, 99: 119, 100: 2, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 107: 9, 108: 10, 109: 11, 110: 13, 111: 14, 112: 15, 113: 16, 114: 17, 115: 18, 116: 19, 117: 20, 118: 21, 119: 22, 120: 24}
    mydict = {0: 1, 1: 10, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106, 9: 107, 10: 108, 11: 109, 12: 11, 13: 110, 14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 23: 12, 24: 120, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18, 31: 19, 32: 2, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26, 40: 27, 41: 28, 42: 29, 43: 3, 44: 30, 45: 31, 46: 32, 47: 33, 48: 34, 49: 35, 50: 36, 51: 37, 52: 38, 53: 39, 54: 4, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44, 60: 45, 61: 46, 62: 47, 63: 48, 64: 49, 65: 5, 66: 50, 67: 51, 68: 52, 69: 53, 70: 54, 71: 55, 72: 56, 73: 57, 74: 58, 75: 59, 76: 6, 77: 60, 78: 61, 79: 62, 80: 63, 81: 64, 82: 65, 83: 66, 84: 67, 85: 68, 86: 69, 87: 7, 88: 70, 89: 71, 90: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77, 96: 78, 97: 79, 98: 8, 99: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 105: 86, 106: 87, 107: 88, 108: 89, 109: 9, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96, 117: 97, 118: 98, 119: 99}
    use_gpu = True
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    target_label=np.array([src_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    target_label2=np.array([src_label2]).astype('int64')
    target_label2=np.expand_dims(target_label2, axis=0)
    
    img = o.copy()
    # decay_factor = 0.94
    # steps=80
    # epsilons = np.linspace(1.5, 388, num=45)
    # decay_factor = 0.94
    # steps=80
    # epsilons = np.linspace(3.5, 388, num=45)   89.27
    # decay_factor = 0.94
    #tmp = np.percentile(abs(norm_m), [25, 50, 99.5])
    # steps=100
    # epsilons = np.linspace(5, 388, num=45)    91.53
    ##################################################
    # decay_factor = 0.94
    # steps=100
    # epsilons = np.linspace(5, 388, num=45)     决赛83.39
    decay_factor = 0.94
    steps=90
    epsilons = np.linspace(5, 388, num=65)
    for epsilon in epsilons[:]:
        momentum = 0
        adv=img.copy()
        
        for i in range(steps):
            if i<50:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.1,size = (3,224,224))).astype('float32')
            else:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.05,size = (3,224,224))).astype('float32')
                
            g,resul1,resul2,resul3,resul4,resul5,resul6,resul7,resul8 = exe.run(adv_program,
                         fetch_list=[gradients,out1,out2,out3,out4,out5,out6,out7,out8],
                         feed={'label2':target_label2,'adv_image':adv_noise,'label': target_label }
                          )
            #print(g.shape)
            g = g[0][0]
            #print(g.shape)
            velocity = g / (np.linalg.norm(g.flatten(),ord=1) + 1e-10)
            momentum = decay_factor * momentum + velocity
            #print(momentum.shape)
            norm_m = momentum / (np.linalg.norm(momentum.flatten(),ord=2) + 1e-10)
            #print(norm_m.shape)
            _max = np.max(abs(norm_m))
            tmp = np.percentile(abs(norm_m), [25, 50, 99.5])
            thres = tmp[2]
            mask = abs(norm_m)>thres
            norm_m_m = np.multiply(norm_m,mask)
            #print(norm_m_m.shape)
            
            if i==0:
                adv=adv+epsilon*norm_m_m 
            else:
                adv=adv-epsilon*norm_m_m #这里用的是没有mask的
            #实施linf约束
            adv=linf_img_tenosr(img,adv,epsilon)

        print("epsilon is {}".format(epsilon))
        print("true label:{}; model1:{}; model2:{};model3:{};model4:{};model5:{}; model6:{}; model7:{}; model8:{} ".format(label,resul1.argmax(),resul2.argmax(),mydict[resul3.argmax()],mydict[resul4.argmax()],\
        mydict[resul5.argmax()],mydict[resul6.argmax()],mydict[resul7.argmax()],mydict[resul8.argmax()]))#模型3标签到真正标签
        
        if((label!=resul1.argmax()) and(label!=resul2.argmax())and(origdict[label]!=resul3.argmax())and(origdict[label]!=resul4.argmax())and(origdict[label]!=resul5.argmax())\
        and(origdict[label]!=resul6.argmax())and(origdict[label]!=resul7.argmax())and(origdict[label]!=resul8.argmax())):
            break
    return adv
    
def ensem_mom_attack_threshold_9model(adv_program,eval_program,gradients,o,src_label2,src_label,out1,out2,out3,out4,out5,out6,out7,out8,out9,label,iteration=20,use_gpu = True):
    #将小扰动调整为0
    origdict = {1: 0, 2: 32, 3: 43, 4: 54, 5: 65, 6: 76, 7: 87, 8: 98, 9: 109, 10: 1, 11: 12, 12: 23, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29, 18: 30, 19: 31, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 28: 41, 29: 42, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48, 35: 49, 36: 50, 37: 51, 38: 52, 39: 53, 40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 60, 46: 61, 47: 62, 48: 63, 49: 64, 50: 66, 51: 67, 52: 68, 53: 69, 54: 70, 55: 71, 56: 72, 57: 73, 58: 74, 59: 75, 60: 77, 61: 78, 62: 79, 63: 80, 64: 81, 65: 82, 66: 83, 67: 84, 68: 85, 69: 86, 70: 88, 71: 89, 72: 90, 73: 91, 74: 92, 75: 93, 76: 94, 77: 95, 78: 96, 79: 97, 80: 99, 81: 100, 82: 101, 83: 102, 84: 103, 85: 104, 86: 105, 87: 106, 88: 107, 89: 108, 90: 110, 91: 111, 92: 112, 93: 113, 94: 114, 95: 115, 96: 116, 97: 117, 98: 118, 99: 119, 100: 2, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 107: 9, 108: 10, 109: 11, 110: 13, 111: 14, 112: 15, 113: 16, 114: 17, 115: 18, 116: 19, 117: 20, 118: 21, 119: 22, 120: 24}
    mydict = {0: 1, 1: 10, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106, 9: 107, 10: 108, 11: 109, 12: 11, 13: 110, 14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 23: 12, 24: 120, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18, 31: 19, 32: 2, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26, 40: 27, 41: 28, 42: 29, 43: 3, 44: 30, 45: 31, 46: 32, 47: 33, 48: 34, 49: 35, 50: 36, 51: 37, 52: 38, 53: 39, 54: 4, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44, 60: 45, 61: 46, 62: 47, 63: 48, 64: 49, 65: 5, 66: 50, 67: 51, 68: 52, 69: 53, 70: 54, 71: 55, 72: 56, 73: 57, 74: 58, 75: 59, 76: 6, 77: 60, 78: 61, 79: 62, 80: 63, 81: 64, 82: 65, 83: 66, 84: 67, 85: 68, 86: 69, 87: 7, 88: 70, 89: 71, 90: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77, 96: 78, 97: 79, 98: 8, 99: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 105: 86, 106: 87, 107: 88, 108: 89, 109: 9, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96, 117: 97, 118: 98, 119: 99}
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    target_label=np.array([src_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    target_label2=np.array([src_label2]).astype('int64')
    target_label2=np.expand_dims(target_label2, axis=0)
    
    img = o.copy()
    decay_factor = 0.90
    steps=90
    epsilons = np.linspace(5, 388, num=75)
    for epsilon in epsilons[:]:
        momentum = 0
        adv=img.copy()
        
        for i in range(steps):
            #if i <3:
            #    adv_noise = (adv+np.random.normal(loc=0.0, scale=1.5+epsilon/90,size = (3,224,224))).astype('float32')
            if i<50:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.5+epsilon/90,size = (3,224,224))).astype('float32')
            else:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.1,size = (3,224,224))).astype('float32')
                
            g,resul1,resul2,resul3,resul4,resul5,resul6,resul7,resul8,resul9 = exe.run(adv_program,
                         fetch_list=[gradients,out1,out2,out3,out4,out5,out6,out7,out8,out9],
                         feed={'label2':target_label2,'adv_image':adv_noise,'label': target_label }
                          )
            #print(g[0][0].shape,g[0][1].shape,g[0][2].shape)
            g = (g[0][0]+g[0][1]+g[0][2])/3 #三通道梯度平均
            #print(g.shape)
            velocity = g / (np.linalg.norm(g.flatten(),ord=1) + 1e-10)
            momentum = decay_factor * momentum + velocity
            #print(momentum.shape)
            norm_m = momentum / (np.linalg.norm(momentum.flatten(),ord=2) + 1e-10)
            #print(norm_m.shape)
            _max = np.max(abs(norm_m))
            tmp = np.percentile(abs(norm_m), [25, 99.45, 99.5])#将图片变动的像素点限定在0.5%
            thres = tmp[1]
            mask = abs(norm_m)>thres
            norm_m_m = np.multiply(norm_m,mask)
            #print(norm_m_m.shape)
            
            if i==0:
                adv=adv+epsilon*norm_m_m 
            else:
                adv=adv-epsilon*norm_m_m 
            #实施linf约束
            adv=linf_img_tenosr(img,adv,epsilon)

        print("epsilon is {}".format(epsilon))
        print("true label is:{}; model1:{}; model2:{}; model3:{}; model4:{}; model5:{}; model6:{}; model7:{}; model8:{} ; model9:{} ".format(label,resul1.argmax(),resul2.argmax(),mydict[resul3.argmax()],mydict[resul4.argmax()],\
        mydict[resul5.argmax()],mydict[resul6.argmax()],mydict[resul7.argmax()],mydict[resul8.argmax()],mydict[resul9.argmax()]))#模型3标签到真正标签

        if((label!=resul1.argmax()) and(label!=resul2.argmax())and(origdict[label]!=resul3.argmax())and(origdict[label]!=resul4.argmax())and(origdict[label]!=resul5.argmax())\
        and(origdict[label]!=resul6.argmax())and(origdict[label]!=resul7.argmax())and(origdict[label]!=resul8.argmax())and(origdict[label]!=resul9.argmax())):
            break
    return adv

def ensem_mom_attack_threshold_9model_tarversion(adv_program,eval_program,gradients,o,src_label2,src_label,out1,out2,out3,out4,out5,out6,out7,out8,out9,label,mm,iteration=20,use_gpu = True):
    #将小扰动调整为0
    #tarversion旨在迭代最后两次，进行target梯度下降
    origdict = {1: 0, 2: 32, 3: 43, 4: 54, 5: 65, 6: 76, 7: 87, 8: 98, 9: 109, 10: 1, 11: 12, 12: 23, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29, 18: 30, 19: 31, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 28: 41, 29: 42, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48, 35: 49, 36: 50, 37: 51, 38: 52, 39: 53, 40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 60, 46: 61, 47: 62, 48: 63, 49: 64, 50: 66, 51: 67, 52: 68, 53: 69, 54: 70, 55: 71, 56: 72, 57: 73, 58: 74, 59: 75, 60: 77, 61: 78, 62: 79, 63: 80, 64: 81, 65: 82, 66: 83, 67: 84, 68: 85, 69: 86, 70: 88, 71: 89, 72: 90, 73: 91, 74: 92, 75: 93, 76: 94, 77: 95, 78: 96, 79: 97, 80: 99, 81: 100, 82: 101, 83: 102, 84: 103, 85: 104, 86: 105, 87: 106, 88: 107, 89: 108, 90: 110, 91: 111, 92: 112, 93: 113, 94: 114, 95: 115, 96: 116, 97: 117, 98: 118, 99: 119, 100: 2, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 107: 9, 108: 10, 109: 11, 110: 13, 111: 14, 112: 15, 113: 16, 114: 17, 115: 18, 116: 19, 117: 20, 118: 21, 119: 22, 120: 24}
    mydict = {0: 1, 1: 10, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106, 9: 107, 10: 108, 11: 109, 12: 11, 13: 110, 14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 23: 12, 24: 120, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18, 31: 19, 32: 2, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26, 40: 27, 41: 28, 42: 29, 43: 3, 44: 30, 45: 31, 46: 32, 47: 33, 48: 34, 49: 35, 50: 36, 51: 37, 52: 38, 53: 39, 54: 4, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44, 60: 45, 61: 46, 62: 47, 63: 48, 64: 49, 65: 5, 66: 50, 67: 51, 68: 52, 69: 53, 70: 54, 71: 55, 72: 56, 73: 57, 74: 58, 75: 59, 76: 6, 77: 60, 78: 61, 79: 62, 80: 63, 81: 64, 82: 65, 83: 66, 84: 67, 85: 68, 86: 69, 87: 7, 88: 70, 89: 71, 90: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77, 96: 78, 97: 79, 98: 8, 99: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 105: 86, 106: 87, 107: 88, 108: 89, 109: 9, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96, 117: 97, 118: 98, 119: 99}
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    target_label=np.array([src_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    target_label2=np.array([src_label2]).astype('int64')
    target_label2=np.expand_dims(target_label2, axis=0)
    
    img = o.copy()
    decay_factor = 0.90
    steps=90
    epsilons = np.linspace(5, 388, num=75)
    flag_traget = 0#表示非目标攻击
    flag2=0 #退出的标志
    for epsilon in epsilons[:]:
        #print("now momentum is {}".format(momentum))
        if flag_traget==0:
            #momentum = mm
            momentum = 0
            adv=img.copy()
            for i in range(steps):
                
                if i<50:
                    adv_noise = (adv+np.random.normal(loc=0.0, scale=0.5+epsilon/90,size = (3,224,224))).astype('float32')
                else:
                    adv_noise = (adv+np.random.normal(loc=0.0, scale=0.1,size = (3,224,224))).astype('float32')
                g,resul1,resul2,resul3,resul4,resul5,resul6,resul7,resul8,resul9 = exe.run(adv_program,
                             fetch_list=[gradients,out1,out2,out3,out4,out5,out6,out7,out8,out9],
                             feed={'label2':target_label2,'adv_image':adv_noise,'label': target_label })
               
                #print(g[0][0].shape,g[0][1].shape,g[0][2].shape)
                g = (g[0][0]+g[0][1]+g[0][2])/3 #三通道梯度平均
                #print(g.shape)
                velocity = g / (np.linalg.norm(g.flatten(),ord=1) + 1e-10)
                momentum = decay_factor * momentum + velocity
                #print(momentum.shape)
                norm_m = momentum / (np.linalg.norm(momentum.flatten(),ord=2) + 1e-10)
                #print(norm_m.shape)
                _max = np.max(abs(norm_m))
                tmp = np.percentile(abs(norm_m), [25, 99.45, 99.5])#将图片变动的像素点限定在0.5%
                thres = tmp[2]
                mask = abs(norm_m)>thres
                norm_m_m = np.multiply(norm_m,mask)
                if i<50: #前50步，2%的梯度反响,随着i递减   试试5%
                    dir_mask = np.random.rand(3,224,224)
                    #print(dir_mask)
                    dir_mask = dir_mask>(0.15-i/900)  
                    #print(dir_mask)
                    dir_mask[dir_mask==0] = -1
                    #print(dir_mask)
                    norm_m_m = np.multiply(norm_m_m,dir_mask)
                    #print(norm_m_m.shape)
                #步长也随着step衰减
                if i==0:
                    adv=adv+epsilon*norm_m_m 
                else:
                    adv=adv-epsilon*norm_m_m 
                    #adv=adv-(epsilon-i/30)*norm_m_m 
                #实施linf约束
                adv=linf_img_tenosr(img,adv,epsilon)
        else:
            for i in range(2):
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.1,size = (3,224,224))).astype('float32')
                target_label=np.array([t_label]).astype('int64')
                target_label=np.expand_dims(target_label, axis=0)
                target_label2=np.array([origdict[t_label]]).astype('int64')
                target_label2=np.expand_dims(target_label2, axis=0)
                g,resul1,resul2,resul3,resul4,resul5,resul6,resul7,resul8,resul9 = exe.run(adv_program,
                         fetch_list=[gradients,out1,out2,out3,out4,out5,out6,out7,out8,out9],
                         feed={'label2':target_label2,'adv_image':adv_noise,'label': target_label }
                          )
                g = (g[0][0]+g[0][1]+g[0][2])/3 #三通道梯度平均
                velocity = g / (np.linalg.norm(g.flatten(),ord=1) + 1e-10)
                momentum = decay_factor * momentum + velocity
                #print(momentum.shape)
                norm_m = momentum / (np.linalg.norm(momentum.flatten(),ord=2) + 1e-10)
                #print(norm_m.shape)
                _max = np.max(abs(norm_m))
                tmp = np.percentile(abs(norm_m), [25, 99.45, 99.5])#将图片变动的像素点限定在0.5%
                thres = tmp[2]
                mask = abs(norm_m)>thres
                norm_m_m = np.multiply(norm_m,mask)
                adv=adv+epsilon*norm_m_m
                #实施linf约束
                adv=linf_img_tenosr(img,adv,epsilon)
            flag2=1
            
        print("epsilon is {}".format(epsilon))
        print("label is:{}; model1:{}; model2:{}; model3:{}; model4:{}; model5:{}; model6:{}; model7:{}; model8:{} ; model9:{} ".format(label,resul1.argmax(),resul2.argmax(),mydict[resul3.argmax()],mydict[resul4.argmax()],\
        mydict[resul5.argmax()],mydict[resul6.argmax()],mydict[resul7.argmax()],mydict[resul8.argmax()],mydict[resul9.argmax()]))#模型3标签到真正标签
        

        if((label!=resul1.argmax()) and(label!=resul2.argmax())and(origdict[label]!=resul3.argmax())and(origdict[label]!=resul4.argmax())and(origdict[label]!=resul5.argmax())\
        and(origdict[label]!=resul6.argmax())and(origdict[label]!=resul7.argmax())and(origdict[label]!=resul8.argmax())and(origdict[label]!=resul9.argmax())):
            res_list = [resul1.argmax(),resul2.argmax(),mydict[resul3.argmax()],mydict[resul4.argmax()],mydict[resul5.argmax()],mydict[resul6.argmax()],mydict[resul7.argmax()],mydict[resul8.argmax()],mydict[resul9.argmax()]]
            ser = pd.Series(res_list)
            t_label = ser.mode()[0]#取众数作为target_label
            flag_traget=1
            if(flag2 == 1):
                break
    return adv,momentum
  
#用于计算原标签排序
def rank_ans(resul1,label):
    #print("resul is {}".format(resul1))
    #print(resul1.shape)
    #print(resul1[0].shape)
    resul1 = resul1[0]
    p_t = resul1[label]
    s_res = sorted(resul1,reverse = True)
    rank = s_res.index(p_t)
    return rank
def ensem_mom_attack_threshold_9model2(adv_program,eval_program,gradients,o,src_label2,src_label,out1,out2,out3,out4,out5,out6,out7,out8,out9,label,iteration=20,use_gpu = True):
    #测试将原排序靠后思路
    origdict = {1: 0, 2: 32, 3: 43, 4: 54, 5: 65, 6: 76, 7: 87, 8: 98, 9: 109, 10: 1, 11: 12, 12: 23, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29, 18: 30, 19: 31, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 28: 41, 29: 42, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48, 35: 49, 36: 50, 37: 51, 38: 52, 39: 53, 40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 60, 46: 61, 47: 62, 48: 63, 49: 64, 50: 66, 51: 67, 52: 68, 53: 69, 54: 70, 55: 71, 56: 72, 57: 73, 58: 74, 59: 75, 60: 77, 61: 78, 62: 79, 63: 80, 64: 81, 65: 82, 66: 83, 67: 84, 68: 85, 69: 86, 70: 88, 71: 89, 72: 90, 73: 91, 74: 92, 75: 93, 76: 94, 77: 95, 78: 96, 79: 97, 80: 99, 81: 100, 82: 101, 83: 102, 84: 103, 85: 104, 86: 105, 87: 106, 88: 107, 89: 108, 90: 110, 91: 111, 92: 112, 93: 113, 94: 114, 95: 115, 96: 116, 97: 117, 98: 118, 99: 119, 100: 2, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 107: 9, 108: 10, 109: 11, 110: 13, 111: 14, 112: 15, 113: 16, 114: 17, 115: 18, 116: 19, 117: 20, 118: 21, 119: 22, 120: 24}
    mydict = {0: 1, 1: 10, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106, 9: 107, 10: 108, 11: 109, 12: 11, 13: 110, 14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 23: 12, 24: 120, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18, 31: 19, 32: 2, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26, 40: 27, 41: 28, 42: 29, 43: 3, 44: 30, 45: 31, 46: 32, 47: 33, 48: 34, 49: 35, 50: 36, 51: 37, 52: 38, 53: 39, 54: 4, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44, 60: 45, 61: 46, 62: 47, 63: 48, 64: 49, 65: 5, 66: 50, 67: 51, 68: 52, 69: 53, 70: 54, 71: 55, 72: 56, 73: 57, 74: 58, 75: 59, 76: 6, 77: 60, 78: 61, 79: 62, 80: 63, 81: 64, 82: 65, 83: 66, 84: 67, 85: 68, 86: 69, 87: 7, 88: 70, 89: 71, 90: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77, 96: 78, 97: 79, 98: 8, 99: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 105: 86, 106: 87, 107: 88, 108: 89, 109: 9, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96, 117: 97, 118: 98, 119: 99}
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    target_label=np.array([src_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    target_label2=np.array([src_label2]).astype('int64')
    target_label2=np.expand_dims(target_label2, axis=0)
    
    img = o.copy()
    decay_factor = 0.94
    steps=90
    #epsilons = np.linspace(6, 388, num=75)
    epsilons = [5,6,8,10,15,25,30,50]
    for epsilon in epsilons[:]:
        momentum = 0
        adv=img.copy()
        
        for i in range(steps):
            if i<50:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.6+epsilon/90,size = (3,224,224))).astype('float32')
            else:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.15,size = (3,224,224))).astype('float32')
                
            g,resul1,resul2,resul3,resul4,resul5,resul6,resul7,resul8,resul9 = exe.run(adv_program,
                         fetch_list=[gradients,out1,out2,out3,out4,out5,out6,out7,out8,out9],
                         feed={'label2':target_label2,'adv_image':adv_noise,'label': target_label }
                          )
            #print(g[0][0].shape,g[0][1].shape,g[0][2].shape)
            g = (g[0][0]+g[0][1]+g[0][2])/3 #三通道梯度平均
            #print(g.shape)
            velocity = g / (np.linalg.norm(g.flatten(),ord=1) + 1e-10)
            momentum = decay_factor * momentum + velocity
            #print(momentum.shape)
            norm_m = momentum / (np.linalg.norm(momentum.flatten(),ord=2) + 1e-10)
            #print(norm_m.shape)
            _max = np.max(abs(norm_m))
            tmp = np.percentile(abs(norm_m), [25, 99.45, 99.5])#将图片变动的像素点限定在0.5%
            thres = tmp[1]
            mask = abs(norm_m)>thres
            norm_m_m = np.multiply(norm_m,mask)
            #print(norm_m_m.shape)
            
            if i==0:#第一步进行梯度上升
                adv=adv+epsilon*norm_m_m 
            else:
                adv=adv-epsilon*norm_m_m 
            #实施linf约束
            adv=linf_img_tenosr(img,adv,epsilon)

        label2 = target_label2[0][0]
        rank1 = rank_ans(resul1,label)
        rank2 = rank_ans(resul2,label)
        rank3 = rank_ans(resul3,label2)
        rank4 = rank_ans(resul4,label2)
        rank5 = rank_ans(resul5,label2)
        rank6 = rank_ans(resul6,label2)
        rank7 = rank_ans(resul7,label2)
        rank8 = rank_ans(resul8,label2)
        rank9 = rank_ans(resul9,label2)
        print("epsilon is {}".format(epsilon))
        print("true label is:{}; model1:{},index is{}; model2:{},index is{}; model3:{},index is{}; model4:{},index is{}; model5:{},index is{}; model6:{},index is{}; model7:{},index is{}; model8:{},index is{}; model9:{},index is{}; "\
        .format(label,\
        resul1.argmax(),rank1, \
        resul2.argmax(),rank2,\
        mydict[resul3.argmax()],rank3,\
        mydict[resul4.argmax()],rank4,\
        mydict[resul5.argmax()],rank5,\
        mydict[resul6.argmax()],rank6,\
        mydict[resul7.argmax()],rank7,\
        mydict[resul8.argmax()],rank8,\
        mydict[resul9.argmax()],rank9))#模型3标签到真正标签
        
        #更改终止条件
        #第一种 都攻击成功就终止
        #if((label!=resul1.argmax()) and(label!=resul2.argmax())and(origdict[label]!=resul3.argmax())and(origdict[label]!=resul4.argmax())and(origdict[label]!=resul5.argmax())\
        #and(origdict[label]!=resul6.argmax())and(origdict[label]!=resul7.argmax())and(origdict[label]!=resul8.argmax())and(origdict[label]!=resul9.argmax())):
        #第二种 原标签排序和大于一个阈值
        #if((np.argwhere(resul1==label)+np.argwhere(resul2==label)+np.argwhere(resul3==label2)+np.argwhere(resul4==label2)+np.argwhere(resul5==label2)+np.argwhere(resul6==label2)+np.argwhere(resul7=label2)+\
        #np.argwhere(resul8==label2)+np.argwhere(resul9==label2))>90):
        
        #第三种 原标签排序都大于min_index
        min_index = 10
        if((rank1>min_index) and (rank2>min_index) and (rank3>min_index)and (rank4>min_index)and (rank5>min_index)\
        and (rank6>min_index)and (rank7>min_index)and (rank8>min_index)and (rank9>min_index)):
            break
    return adv



def ensem_mom_attack_threshold_11model(adv_program,eval_program,gradients,o,src_label2,src_label,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,label,iteration=20,use_gpu = True):
    #将小扰动调整为0
    origdict = {1: 0, 2: 32, 3: 43, 4: 54, 5: 65, 6: 76, 7: 87, 8: 98, 9: 109, 10: 1, 11: 12, 12: 23, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29, 18: 30, 19: 31, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 28: 41, 29: 42, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48, 35: 49, 36: 50, 37: 51, 38: 52, 39: 53, 40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 60, 46: 61, 47: 62, 48: 63, 49: 64, 50: 66, 51: 67, 52: 68, 53: 69, 54: 70, 55: 71, 56: 72, 57: 73, 58: 74, 59: 75, 60: 77, 61: 78, 62: 79, 63: 80, 64: 81, 65: 82, 66: 83, 67: 84, 68: 85, 69: 86, 70: 88, 71: 89, 72: 90, 73: 91, 74: 92, 75: 93, 76: 94, 77: 95, 78: 96, 79: 97, 80: 99, 81: 100, 82: 101, 83: 102, 84: 103, 85: 104, 86: 105, 87: 106, 88: 107, 89: 108, 90: 110, 91: 111, 92: 112, 93: 113, 94: 114, 95: 115, 96: 116, 97: 117, 98: 118, 99: 119, 100: 2, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 107: 9, 108: 10, 109: 11, 110: 13, 111: 14, 112: 15, 113: 16, 114: 17, 115: 18, 116: 19, 117: 20, 118: 21, 119: 22, 120: 24}
    mydict = {0: 1, 1: 10, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106, 9: 107, 10: 108, 11: 109, 12: 11, 13: 110, 14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 23: 12, 24: 120, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18, 31: 19, 32: 2, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26, 40: 27, 41: 28, 42: 29, 43: 3, 44: 30, 45: 31, 46: 32, 47: 33, 48: 34, 49: 35, 50: 36, 51: 37, 52: 38, 53: 39, 54: 4, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44, 60: 45, 61: 46, 62: 47, 63: 48, 64: 49, 65: 5, 66: 50, 67: 51, 68: 52, 69: 53, 70: 54, 71: 55, 72: 56, 73: 57, 74: 58, 75: 59, 76: 6, 77: 60, 78: 61, 79: 62, 80: 63, 81: 64, 82: 65, 83: 66, 84: 67, 85: 68, 86: 69, 87: 7, 88: 70, 89: 71, 90: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77, 96: 78, 97: 79, 98: 8, 99: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 105: 86, 106: 87, 107: 88, 108: 89, 109: 9, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96, 117: 97, 118: 98, 119: 99}
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    target_label=np.array([src_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    target_label2=np.array([src_label2]).astype('int64')
    target_label2=np.expand_dims(target_label2, axis=0)
    
    img = o.copy()
    decay_factor = 0.94
    steps=90
    epsilons = np.linspace(8, 100, num=35)
    for epsilon in epsilons[:]:
        momentum = 0
        adv=img.copy()
        
        for i in range(steps):
            if i<50:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.8+epsilon/90,size = (3,224,224))).astype('float32')
            else:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.2,size = (3,224,224))).astype('float32')
                
            g,resul1,resul2,resul3,resul4,resul5,resul6,resul7,resul8,resul9,resul10,result11 = exe.run(adv_program,
                         fetch_list=[gradients,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11],
                         feed={'label2':target_label2,'adv_image':adv_noise,'label': target_label }
                          )
            #print(g[0][0].shape,g[0][1].shape,g[0][2].shape)
            g = (g[0][0]+g[0][1]+g[0][2])/3 #三通道梯度平均
            #print(g.shape)
            velocity = g / (np.linalg.norm(g.flatten(),ord=1) + 1e-10)
            momentum = decay_factor * momentum + velocity
            #print(momentum.shape)
            norm_m = momentum / (np.linalg.norm(momentum.flatten(),ord=2) + 1e-10)
            #print(norm_m.shape)
            _max = np.max(abs(norm_m))
            tmp = np.percentile(abs(norm_m), [25, 99.47, 99.5])#将图片变动的像素点限定在0.5%
            thres = tmp[1]
            mask = abs(norm_m)>thres
            norm_m_m = np.multiply(norm_m,mask)
            #print(norm_m_m.shape)
            if i<50: #前50步，2%的梯度反响,随着i递减   试试5%
                dir_mask = np.random.rand(3,224,224)
                #print(dir_mask)
                dir_mask = dir_mask>(0.15-i/900)  
                #print(dir_mask)
                dir_mask[dir_mask==0] = -1
                #print(dir_mask)
                norm_m_m = np.multiply(norm_m_m,dir_mask)
                #print(norm_m_m.shape)
            
            if i==0:
                adv=adv+epsilon*norm_m_m 
            else:
                adv=adv-epsilon*norm_m_m 
            #实施linf约束
            adv=linf_img_tenosr(img,adv,epsilon)

        print("epsilon is {}".format(epsilon))
        print("true label is:{}; model1:{}; model2:{}; model3:{}; model4:{}; model5:{}; model6:{}; model7:{}; model8:{} ; model9:{} ;model10:{};model11:{} ".format(label,resul1.argmax(),resul2.argmax(),mydict[resul3.argmax()],mydict[resul4.argmax()],\
        mydict[resul5.argmax()],mydict[resul6.argmax()],mydict[resul7.argmax()],mydict[resul8.argmax()],mydict[resul9.argmax()],resul10.argmax(),result11.argmax()))#模型3标签到真正标签

        if((label!=resul1.argmax())and(origdict[label]!=resul3.argmax())and(origdict[label]!=resul4.argmax())and(origdict[label]!=resul5.argmax())\
        and(origdict[label]!=resul6.argmax())and(origdict[label]!=resul7.argmax())and(origdict[label]!=resul8.argmax())and(origdict[label]!=resul9.argmax())and(label!=resul10.argmax())and(label!=result11.argmax())):
        #终止条件去掉了第二个模型
            break
    return adv
    
    
def ensem_mom_attack_threshold_10model(adv_program,eval_program,gradients,o,src_label2,src_label,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,label,iteration=20,use_gpu = True):
    #将小扰动调整为0
    origdict = {1: 0, 2: 32, 3: 43, 4: 54, 5: 65, 6: 76, 7: 87, 8: 98, 9: 109, 10: 1, 11: 12, 12: 23, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29, 18: 30, 19: 31, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 28: 41, 29: 42, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48, 35: 49, 36: 50, 37: 51, 38: 52, 39: 53, 40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 60, 46: 61, 47: 62, 48: 63, 49: 64, 50: 66, 51: 67, 52: 68, 53: 69, 54: 70, 55: 71, 56: 72, 57: 73, 58: 74, 59: 75, 60: 77, 61: 78, 62: 79, 63: 80, 64: 81, 65: 82, 66: 83, 67: 84, 68: 85, 69: 86, 70: 88, 71: 89, 72: 90, 73: 91, 74: 92, 75: 93, 76: 94, 77: 95, 78: 96, 79: 97, 80: 99, 81: 100, 82: 101, 83: 102, 84: 103, 85: 104, 86: 105, 87: 106, 88: 107, 89: 108, 90: 110, 91: 111, 92: 112, 93: 113, 94: 114, 95: 115, 96: 116, 97: 117, 98: 118, 99: 119, 100: 2, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 107: 9, 108: 10, 109: 11, 110: 13, 111: 14, 112: 15, 113: 16, 114: 17, 115: 18, 116: 19, 117: 20, 118: 21, 119: 22, 120: 24}
    mydict = {0: 1, 1: 10, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106, 9: 107, 10: 108, 11: 109, 12: 11, 13: 110, 14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 23: 12, 24: 120, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18, 31: 19, 32: 2, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26, 40: 27, 41: 28, 42: 29, 43: 3, 44: 30, 45: 31, 46: 32, 47: 33, 48: 34, 49: 35, 50: 36, 51: 37, 52: 38, 53: 39, 54: 4, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44, 60: 45, 61: 46, 62: 47, 63: 48, 64: 49, 65: 5, 66: 50, 67: 51, 68: 52, 69: 53, 70: 54, 71: 55, 72: 56, 73: 57, 74: 58, 75: 59, 76: 6, 77: 60, 78: 61, 79: 62, 80: 63, 81: 64, 82: 65, 83: 66, 84: 67, 85: 68, 86: 69, 87: 7, 88: 70, 89: 71, 90: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77, 96: 78, 97: 79, 98: 8, 99: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 105: 86, 106: 87, 107: 88, 108: 89, 109: 9, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96, 117: 97, 118: 98, 119: 99}
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    target_label=np.array([src_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    target_label2=np.array([src_label2]).astype('int64')
    target_label2=np.expand_dims(target_label2, axis=0)
    
    img = o
    decay_factor = 0.94
    steps=90
    epsilons = np.linspace(10, 388, num=15)
    for epsilon in epsilons[:]:
        momentum = 0
        adv=img.copy()
        
        for i in range(steps):
            #if i <3:
            #    adv_noise = (adv+np.random.normal(loc=0.0, scale=1.5+epsilon/90,size = (3,224,224))).astype('float32')
            if i<50:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.6+epsilon/90,size = (3,224,224))).astype('float32')
            else:
                adv_noise = (adv+np.random.normal(loc=0.0, scale=0.1,size = (3,224,224))).astype('float32')
                
            g,resul1,resul2,resul3,resul4,resul5,resul6,resul7,resul8,resul9,resul10 = exe.run(adv_program,
                         fetch_list=[gradients,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10],
                         feed={'label2':target_label2,'adv_image':adv_noise,'label': target_label }
                          )
            #print(g[0][0].shape,g[0][1].shape,g[0][2].shape)
            g = (g[0][0]+g[0][1]+g[0][2])/3 #三通道梯度平均
            #print(g.shape)
            velocity = g / (np.linalg.norm(g.flatten(),ord=1) + 1e-10)
            momentum = decay_factor * momentum + velocity
            #print(momentum.shape)
            norm_m = momentum / (np.linalg.norm(momentum.flatten(),ord=2) + 1e-10)
            #print(norm_m.shape)
            _max = np.max(abs(norm_m))
            tmp = np.percentile(abs(norm_m), [25, 99.45, 99.5])#将图片变动的像素点限定在0.5%
            thres = tmp[1]
            mask = abs(norm_m)>thres
            norm_m_m = np.multiply(norm_m,mask)
            #print(norm_m_m.shape)
            
            if i==0:
                adv=adv+epsilon*norm_m_m 
            else:
                adv=adv-epsilon*norm_m_m 
            #实施linf约束
            adv=linf_img_tenosr(img,adv,epsilon)


        print("true label is:{}; model1:{}; model2:{}; model3:{}; model4:{}; model5:{}; model6:{}; model7:{}; model8:{} ; model9:{} ;model10:{} ".format(label,resul1.argmax(),resul2.argmax(),mydict[resul3.argmax()],mydict[resul4.argmax()],\
        mydict[resul5.argmax()],mydict[resul6.argmax()],mydict[resul7.argmax()],mydict[resul8.argmax()],mydict[resul9.argmax()],resul10.argmax()))#模型3标签到真正标签

        if((label!=resul1.argmax()) and(label!=resul2.argmax())and(origdict[label]!=resul3.argmax())and(origdict[label]!=resul4.argmax())and(origdict[label]!=resul5.argmax())\
        and(origdict[label]!=resul6.argmax())and(origdict[label]!=resul7.argmax())and(origdict[label]!=resul8.argmax())and(origdict[label]!=resul9.argmax())and(label!=resul10.argmax())):
            break
    return adv