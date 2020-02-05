#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import numpy as np
import paddle.fluid as fluid

#加载自定义文件
import models
from attack.attack_pp import FGSM, PGD
from utils import *


#######parse parameters
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('input',            str,   "./input_image/",     "Input directory with images")
add_arg('input2',           str,   "./output_image/",    "attacked image directory with images")
add_arg('output',           str,   "./posopt_output_image/",    "output directory with images")
add_arg('drop_thres',       int,   10,    "drop_thres")

args = parser.parse_args()
print_arguments(args)


######Init args
image_shape = [3,224,224]
class_dim=121
input_dir = args.input
attacked_dir = args.input2
output_dir = args.output
drop_thres = args.drop_thres

val_list = 'val_list.txt'
use_gpu=True


####### Main #######
def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files

def gen_diff():
    original_files = get_original_file(input_dir + val_list)

    for filename, label in original_files:
        image_name, image_ext = filename.split('.')
        img_path = input_dir + filename
        print("Image: {0} ".format(img_path))
        img=process_img(img_path)
        adv_img_path = attacked_dir + image_name+'.png'
        adv=process_img(adv_img_path)
        
        org_img = tensor2img(img)
        adv_img = tensor2img(adv)
        #10/256 以下的扰动全部截断
        diff = abs(org_img-adv_img)<drop_thres   #<10的为1
        diff_max = abs(org_img-adv_img)>=drop_thres  #>=10的为1
        #<10的保留org_img
        tmp1 = np.multiply(org_img,diff)
        #>10的保留adv_img
        tmp2 = np.multiply(adv_img,diff_max)
        final_img = tmp1+tmp2
        
        save_adv_image(final_img, output_dir+image_name+'.png')

if __name__ == '__main__':
    gen_diff()
