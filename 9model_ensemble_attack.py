#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
import paddle.fluid as fluid

import models
from attack.attack_pp import FGSM, PGD,linf_img_tenosr,ensem_mom_attack_threshold_9model,\
ensem_mom_attack_threshold_9model2,ensem_mom_attack_threshold_9model_tarversion
from utils import init_prog, save_adv_image, process_img, tensor2img, calc_mse, add_arguments, print_arguments

image_shape = [3, 224, 224]
class_dim=121
input_dir = "./input_image/"
output_dir = "./output_image/"

model_name1="ResNeXt50_32x4d"
pretrained_model1="./models_parameters/86.45+88.81ResNeXt50_32x4d"

model_name2="MobileNetV2"
pretrained_model2="./models_parameters/MobileNetV2"

model_name4="VGG16"
pretrained_model4="./models_parameters/VGG16"

model_name3="Densenet121"
pretrained_model3="./models_parameters/Densenet121"

model_name5="mnasnet1_0"
pretrained_model5="./models_parameters/mnasnet1_0"

model_name6="wide_resnet"
pretrained_model6="./models_parameters/wide_resnet"

model_name7="googlenet"
pretrained_model7="./models_parameters/googlenet"

model_name8="nas_mobile_net"
pretrained_model8="./models_parameters/nas_mobile_net"

model_name9="alexnet"
pretrained_model9="./models_parameters/alexnet"

val_list = 'val_list.txt'
use_gpu=True

mydict = {0: 1, 1: 10, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106, 9: 107, 10: 108, 11: 109, 12: 11, 13: 110, 14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 23: 12, 24: 120, 25: 13, 26: 14, 27: 15, 28: 16, 29: 17, 30: 18, 31: 19, 32: 2, 33: 20, 34: 21, 35: 22, 36: 23, 37: 24, 38: 25, 39: 26, 40: 27, 41: 28, 42: 29, 43: 3, 44: 30, 45: 31, 46: 32, 47: 33, 48: 34, 49: 35, 50: 36, 51: 37, 52: 38, 53: 39, 54: 4, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44, 60: 45, 61: 46, 62: 47, 63: 48, 64: 49, 65: 5, 66: 50, 67: 51, 68: 52, 69: 53, 70: 54, 71: 55, 72: 56, 73: 57, 74: 58, 75: 59, 76: 6, 77: 60, 78: 61, 79: 62, 80: 63, 81: 64, 82: 65, 83: 66, 84: 67, 85: 68, 86: 69, 87: 7, 88: 70, 89: 71, 90: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77, 96: 78, 97: 79, 98: 8, 99: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 105: 86, 106: 87, 107: 88, 108: 89, 109: 9, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96, 117: 97, 118: 98, 119: 99}
origdict = {1: 0, 2: 32, 3: 43, 4: 54, 5: 65, 6: 76, 7: 87, 8: 98, 9: 109, 10: 1, 11: 12, 12: 23, 13: 25, 14: 26, 15: 27, 16: 28, 17: 29, 18: 30, 19: 31, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 28: 41, 29: 42, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48, 35: 49, 36: 50, 37: 51, 38: 52, 39: 53, 40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 60, 46: 61, 47: 62, 48: 63, 49: 64, 50: 66, 51: 67, 52: 68, 53: 69, 54: 70, 55: 71, 56: 72, 57: 73, 58: 74, 59: 75, 60: 77, 61: 78, 62: 79, 63: 80, 64: 81, 65: 82, 66: 83, 67: 84, 68: 85, 69: 86, 70: 88, 71: 89, 72: 90, 73: 91, 74: 92, 75: 93, 76: 94, 77: 95, 78: 96, 79: 97, 80: 99, 81: 100, 82: 101, 83: 102, 84: 103, 85: 104, 86: 105, 87: 106, 88: 107, 89: 108, 90: 110, 91: 111, 92: 112, 93: 113, 94: 114, 95: 115, 96: 116, 97: 117, 98: 118, 99: 119, 100: 2, 101: 3, 102: 4, 103: 5, 104: 6, 105: 7, 106: 8, 107: 9, 108: 10, 109: 11, 110: 13, 111: 14, 112: 15, 113: 16, 114: 17, 115: 18, 116: 19, 117: 20, 118: 21, 119: 22, 120: 24}

adv_program=fluid.Program()
startup_program = fluid.Program()

new_scope = fluid.Scope()
#完成初始化
with fluid.program_guard(adv_program):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    label = fluid.layers.data(name="label", shape=[1] ,dtype='int64')
    label2 = fluid.layers.data(name="label2", shape=[1] ,dtype='int64')
    adv_image = fluid.layers.create_parameter(name="adv_image",shape=(1,3,224,224),dtype='float32')
    
    model1 = models.__dict__[model_name1]()
    out_logits1 = model1.net(input=adv_image, class_dim=class_dim)
    out1 = fluid.layers.softmax(out_logits1)

    model2 = models.__dict__[model_name2](scale=2.0)
    out_logits2 = model2.net(input=adv_image, class_dim=class_dim)
    out2 = fluid.layers.softmax(out_logits2)

    _input1 = fluid.layers.create_parameter(name="_input_1", shape=(1,3,224,224),dtype='float32')
    
    model3 = models.__dict__[model_name3]()
    input_layer3,out_logits3 = model3.x2paddle_net(input =adv_image )
    out3 = fluid.layers.softmax(out_logits3[0])
    
    model4 = models.__dict__[model_name4]()
    input_layer4,out_logits4 = model4.x2paddle_net(input =adv_image )
    out4 = fluid.layers.softmax(out_logits4[0])


    model5 = models.__dict__[model_name5]()
    input_layer5,out_logits5 = model5.x2paddle_net(input =adv_image )
    out5 = fluid.layers.softmax(out_logits5[0])

    model6 = models.__dict__[model_name6]()
    input_layer6,out_logits6 = model6.x2paddle_net(input =adv_image)
    out6 = fluid.layers.softmax(out_logits6[0])

    model7 = models.__dict__[model_name7]()
    input_layer7,out_logits7 = model7.x2paddle_net(input =adv_image)
    out7 = fluid.layers.softmax(out_logits7[0])

    model8 = models.__dict__[model_name8]()
    input_layer8,out_logits8 = model8.x2paddle_net(input =adv_image)
    out8 = fluid.layers.softmax(out_logits8[0])
    
    
    model9 = models.__dict__[model_name9]()
    input_layer9,out_logits9 = model9.x2paddle_net(input =adv_image)
    out9 = fluid.layers.softmax(out_logits9[0])

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    one_hot_label = fluid.one_hot(input=label, depth=121)
    one_hot_label2 = fluid.one_hot(input=label2, depth=121)
    smooth_label = fluid.layers.label_smooth(label=one_hot_label, epsilon=0.1, dtype="float32")[0]
    #print(smooth_label.shape)
    smooth_label2 = fluid.layers.label_smooth(label=one_hot_label2, epsilon=0.1, dtype="float32")[0]
    #print(smooth_label2.shape)
    #print(one_hot_label)
    #print(smooth_label)
    #尝试三种损失函数
    #第一种
    loss_logp = -1*fluid.layers.log(1-fluid.layers.matmul(out1,one_hot_label[0],transpose_y=True))\
            -1*fluid.layers.log(1-fluid.layers.matmul(out2,one_hot_label[0],transpose_y=True))\
            -1*fluid.layers.log(1-fluid.layers.matmul(out3,one_hot_label2[0],transpose_y=True))\
            -1*fluid.layers.log(1-fluid.layers.matmul(out4,one_hot_label2[0],transpose_y=True))\
            -1*fluid.layers.log(1-fluid.layers.matmul(out5,one_hot_label2[0],transpose_y=True))\
            -1*fluid.layers.log(1-fluid.layers.matmul(out6,one_hot_label2[0],transpose_y=True))\
            -1*fluid.layers.log(1-fluid.layers.matmul(out8,one_hot_label2[0],transpose_y=True))\
            -1*fluid.layers.log(1-fluid.layers.matmul(out9,one_hot_label2[0],transpose_y=True))
    #paddle.fluid.layers.elementwise_min(x, y, axis=-1, act=None, name=None)[源代码]
    #ze =fluid.layers.zeros(shape=[1], dtype='float64')
    
    #第二种
    #without label_smooth
    ze = fluid.layers.fill_constant(shape=[1], value=-1, dtype='float32')
    loss = 1.2*fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out1, label=label[0]))\
    + 0.2*fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out2, label=label[0]))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out3, label=label2[0]))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out4, label=label2[0]))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out5, label=label2[0]))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out6, label=label2[0]))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out7, label=label2[0]))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out8, label=label2[0]))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out9, label=label2[0]))
    
    #with label_smooth
    loss_smooth = 1.2*fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out1, label=smooth_label, soft_label=True))\
    + 0.2*fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out2, label=smooth_label, soft_label=True))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out3, label=smooth_label2, soft_label=True))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out4, label=smooth_label2, soft_label=True))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out5, label=smooth_label2, soft_label=True))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out6, label=smooth_label2, soft_label=True))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out7, label=smooth_label2, soft_label=True))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out8, label=smooth_label2, soft_label=True))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out9, label=smooth_label2, soft_label=True))
    
    
    #第三种
    out_total1 = fluid.layers.softmax(out_logits1[0]+out_logits2[0])
    out_total2 = fluid.layers.softmax(out_logits3[0]+out_logits4[0]+out_logits5[0]+out_logits6[0]+out_logits7[0]+out_logits8[0]+out_logits9[0])
    loss2 = fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out_total1, label=label[0]))\
    + fluid.layers.matmul(ze, fluid.layers.cross_entropy(input=out_total2, label=label2[0]))

    avg_loss=fluid.layers.reshape(loss ,[1])#这里修改loss

init_prog(adv_program)
eval_program = adv_program.clone(for_test=True)

with fluid.program_guard(adv_program): 
    #没有解决变量重名的问题
    def if_exist(var):
        b = os.path.exists(os.path.join(pretrained_model1, var.name))
        return b
    def if_exist2(var):
        b = os.path.exists(os.path.join(pretrained_model2, var.name))
        return b
    def if_exist3(var):
        b = os.path.exists(os.path.join(pretrained_model3, var.name))
        return b
    def if_exist4(var):
        b = os.path.exists(os.path.join(pretrained_model4, var.name))
        return b
    def if_exist5(var):
        b = os.path.exists(os.path.join(pretrained_model5, var.name))
        return b
    def if_exist6(var):
        b = os.path.exists(os.path.join(pretrained_model6, var.name))
        return b
    def if_exist7(var):
        b = os.path.exists(os.path.join(pretrained_model7, var.name))
        return b
    def if_exist8(var):
        b = os.path.exists(os.path.join(pretrained_model8, var.name))
        return b
    def if_exist9(var):
        b = os.path.exists(os.path.join(pretrained_model9, var.name))
        return b
    fluid.io.load_vars(exe,
                       pretrained_model1,
                       fluid.default_main_program(),
                       predicate=if_exist)
    fluid.io.load_vars(exe,
                       pretrained_model2,
                       fluid.default_main_program(),
                       predicate=if_exist2)
    fluid.io.load_vars(exe,
                       pretrained_model3,
                       fluid.default_main_program(),
                       predicate=if_exist3)
    fluid.io.load_vars(exe,
                       pretrained_model4,
                       fluid.default_main_program(),
                       predicate=if_exist4)
    fluid.io.load_vars(exe,
                       pretrained_model5,
                       fluid.default_main_program(),
                       predicate=if_exist5)
    fluid.io.load_vars(exe,
                       pretrained_model6,
                       fluid.default_main_program(),
                       predicate=if_exist6)
    fluid.io.load_vars(exe,
                       pretrained_model7,
                       fluid.default_main_program(),
                       predicate=if_exist7)
    fluid.io.load_vars(exe,
                       pretrained_model8,
                       fluid.default_main_program(),
                       predicate=if_exist8)

    fluid.io.load_vars(exe,
                       pretrained_model9,
                       fluid.default_main_program(),
                       predicate=if_exist9)
    gradients = fluid.backward.gradients(targets=avg_loss, inputs=[adv_image])[0]
    #gradients = fluid.backward.gradients(targets=avg_loss, inputs=[adv_image])
    #print(gradients.shape)
    
def attack_nontarget_by_ensemble(img, src_label,src_label2,label,momentum): #src_label2为转换后的标签
    adv,m=ensem_mom_attack_threshold_9model_tarversion(adv_program=adv_program,eval_program=eval_program,gradients=gradients,o=img,
                src_label = src_label,
                src_label2 = src_label2,
                label = label,
                out1 = out1,out2 = out2 ,out3 = out3 ,out4 = out4,out5 = out5,out6 = out6,out7 = out7 ,out8 = out8,out9 = out9,mm = momentum)#添加了mm

    adv_img=tensor2img(adv)
    return adv_img,m

def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files
    
def gen_adv():
    mse = 0
    original_files = get_original_file(input_dir + val_list)
    #下一个图片的初始梯度方向为上一代的最后的值
    global momentum
    momentum=0
    
    for filename, label in original_files:
        img_path = input_dir + filename
        print("Image: {0} ".format(img_path))
        img=process_img(img_path)
        #adv_img = attack_nontarget_by_ensemble(img, label,origdict[label],label)
        adv_img,m = attack_nontarget_by_ensemble(img, label,origdict[label],label,momentum)
        #m为上一个样本最后一次梯度值
        momentum = m
        #adv_img 已经经过转换了，范围是0-255

        image_name, image_ext = filename.split('.')
        ##Save adversarial image(.png)
        save_adv_image(adv_img, output_dir+image_name+'.png')
        org_img = tensor2img(img)
        score = calc_mse(org_img, adv_img)
        print("Image:{0}, mase = {1} ".format(img_path,score))
        mse += score
    print("ADV {} files, AVG MSE: {} ".format(len(original_files), mse/len(original_files)))

def main():
    gen_adv()

if __name__ == '__main__':
    main()