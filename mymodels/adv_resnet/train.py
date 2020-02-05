import os
import math
import random

import paddle
import paddle.fluid as fluid

import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from resnext import ResNeXt50_32x4d
random.seed(42)
np.random.seed(42)
import sys
# 设置训练和预测时的批大小
batch_size = 32
EPOCH_NUM=20
params_dirname = "./checkpoint"
pretrained_model_path = './ResNeXt50_32x4d/'

# 从预训练模型参数列表获取 ResNet-50 中所有持久参数列表
params_list = os.listdir(pretrained_model_path)

# 预训练数据集 ImageNet 所有图像的均值与方差，用于输入图像标准化
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

# 正则化一张图片
def normalize_image(image):
    image = (image - mean) / std
    return image

# 逆向正则化一张图片
def denorm_image(image):
    return image * std + mean


pretrained_model_path = './ResNeXt50_32x4d/'

# 从预训练模型参数列表获取 ResNet-50 中所有持久参数列表
params_list = os.listdir(pretrained_model_path)
label_names = [str(i) for i in range(1,121)]

def list_files(basepath, mode='train'):
    assert mode in ['train', 'test', 'val']
    
    files = []
    for name in label_names:
        class_files = sorted(os.listdir(os.path.join(basepath, mode,name)))
        for file in class_files:
            files.append((os.path.join(basepath, mode,name, file), int(name)))
    return files

class BaseReader(object):
    """定义 PaddlePaddle 的 Reader 的生成类
    
    由它提供生成 train_reader 和 test_reader 两种类型的 Reader
    使用两种类型的 reader 的原因是因为 train 和 test 在图像预处理 (图像增强) 的部分使用的操作不相同
    
    在训练的过程中，尽可能希望模型能够看到更多更丰富更多样化的输入数据，所以经常会使用类似于
    * 随机切块
    * 随机翻转
    * 随机调整图像明度、色彩
    等等操作。
    
    而预测/验证的情况下，尽可能要保证图像原本完整的信息，可能会使用的有
    * 图像拉伸
    * 图像中心切块
    等等尽量能够保留最多最显著图像内容部分的操作。
    """
    @staticmethod
    def rescale_image(img, target_size):
        width, height = img.size
        percent = float(target_size) / min(width, height)
        resized_width = int(round(width * percent))
        resized_height = int(round(height * percent))
        img = img.resize((resized_width, resized_height), PIL.Image.LANCZOS)
        return img
    
    @staticmethod
    def resize_image(img, target_size):
        img = img.resize((target_size, target_size), PIL.Image.LANCZOS)
        return img

    @staticmethod
    def crop_image(img, target_size, center=True):
        width, height = img.size
        size = target_size
        if center == True:
            w_start = (width - size) / 2
            h_start = (height - size) / 2
        else:
            w_start = np.random.randint(0, width - size + 1)
            h_start = np.random.randint(0, height - size + 1)
        w_end = w_start + size
        h_end = h_start + size
        img = img.crop((w_start, h_start, w_end, h_end))
        return img
    
    @staticmethod
    def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
        aspect_ratio = math.sqrt(np.random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                    (float(img.size[1]) / img.size[0]) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = np.random.randint(0, img.size[0] - w + 1)
        j = np.random.randint(0, img.size[1] - h + 1)

        img = img.crop((i, j, i + w, j + h))
        img = img.resize((size, size), PIL.Image.LANCZOS)
        return img
    
    @staticmethod
    def random_flip(img):
        if np.random.randint(0, 2) == 1:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def rotate_image(img):
        angle = np.random.randint(-10, 11)
        img = img.rotate(angle)
        return img
    
    def create_train_reader(self, files):
        def _reader():
            for file, labelname in files:
                image = PIL.Image.open(file)
                image = BaseReader.resize_image(image, 224)
                #image = BaseReader.rotate_image(image)
                image = BaseReader.crop_image(image, 224)
                #image = BaseReader.random_flip(image)
                
                image = np.array(image).astype('float32').transpose((2, 0, 1)) / 255
                image = normalize_image(image)
                
                label = labelname
                yield image, label
        return _reader

    def create_test_reader(self, files):
        def _reader():
            for file, labelname in files:
                image = PIL.Image.open(file)
                
                image = BaseReader.resize_image(image, 224)
                image = BaseReader.crop_image(image, 224)
                
                image = np.array(image).astype('float32').transpose((2, 0, 1)) / 255
                image = normalize_image(image)
                
                label = labelname
                yield image, label
        return _reader


def create_network(image, label, class_dim=121, is_test=False):
    # image 以及 label 是 Variable 类型，在再训练过程中，他们是在 py_reader 的数据流中产生的
    # 其类型和大小如下：
    #   image = fluid.layers.data(name='image', shape=(-1, 3, 224, 224), dtype='float32')
    #   label = fluid.layers.data(name='label', shape=(-1, 1), dtype='int64')
    
    # is_test 是表明现在是在创建训练还是验证模型
    # 由于是两个 fluid.Program 上下文，故建议分别创建
    
    model = ResNeXt50_32x4d()
    out = model.net(image, class_dim=class_dim)
    
    confidence = fluid.layers.softmax(out)
    top5_scores, top5_indices = fluid.layers.topk(confidence, k=5)
    
    loss = fluid.layers.cross_entropy(input=confidence, label=label)
    loss = fluid.layers.mean(x=loss)
    
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    
    fetch_list = [
        image.name, label.name,
        out.name, confidence.name, top5_scores.name, top5_indices.name,
        loss.name, acc_top1.name, acc_top5.name
    ] 
    return loss, fetch_list

files_path = './data'


train_files = list_files(files_path, mode='train')
print(len(train_files))
test_files = list_files(files_path, mode='test')
np.random.shuffle(train_files)
np.random.shuffle(test_files)

def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=0.001)

def inference_network():
    image = fluid.layers.data(name='image', shape=(None, 3, 224, 224), dtype='float32')
#         label = fluid.layers.data(name='label', shape=(-1, 1), dtype='int64'
    model = ResNeXt50_32x4d()
    out = model.net(image, class_dim=121)
    predict = fluid.layers.softmax(out)
    return predict

def train_network(predict):
    #predict = inference_program()
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]

def train():
    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    BATCH_SIZE = 128
    base_reader = BaseReader()

    # 重新分别创建训练使用的 reader 和 预测使用的 reader
    base_train_reader = base_reader.create_train_reader(train_files)
    train_reader = paddle.batch(base_train_reader, batch_size=batch_size)
    base_test_reader = base_reader.create_test_reader(test_files)
    test_reader = paddle.batch(base_test_reader, batch_size=batch_size)
    
    feed_order = ['image', 'label']
    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()
    
    predict = inference_network()
    avg_cost, acc = train_network(predict)

    test_program = main_program.clone(for_test=True)
    optimizer = optimizer_program()
    optimizer.minimize(avg_cost)
    
    exe = fluid.Executor(place)

    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len([avg_cost, acc]) * [0]
        for tid, test_data in enumerate(reader()):
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=[avg_cost, acc])
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]
# main train loop.
    def train_loop():
        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(start_program)
        def _predicate(var):
        # 查看 var 参数是不是在预训练模型路径内
            return os.path.exists(os.path.join(pretrained_model_path, var.name))
        fluid.io.load_vars(exe, pretrained_model_path, predicate=_predicate, main_program=test_program)
        print("pretrained model loaded")
        step = 0
        for pass_id in range(EPOCH_NUM):
            loss_ep = []
            acc_ep = []
            total=0
            for step_id, data_train in enumerate(train_reader()):
                total +=len(data_train)
                #print(total)
                #print(len(data_train))
                avg_loss_value = exe.run(
                        main_program,
                        feed=feeder.feed(data_train),
                        fetch_list=[avg_cost, acc])
                sys.stdout.write('.')
                sys.stdout.flush()
                step += 1
                    #print(avg_loss_value[0][0])
                    #print(avg_loss_value[1][0])
                loss_ep.append(avg_loss_value[0][0])
                acc_ep.append(avg_loss_value[1][0])
                    #if step_id == 3:
                    #    print("train epo {},loss = {} ,acc = {}".format(step_id,np.mean(loss_ep),np.mean(acc_ep)))
            print("train epo {},loss = {} ,acc = {}".format(step_id,np.mean(loss_ep),np.mean(acc_ep)))
            
            avg_cost_test, accuracy_test = train_test(
                test_program, reader=test_reader)
            print('\nTest with Pass {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
                pass_id, avg_cost_test, accuracy_test))

            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["image"],
                                              [predict], exe)

            if 1 and pass_id == EPOCH_NUM - 1:
                print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
                print("kpis\ttrain_acc\t%f" % avg_loss_value[1])
                print("kpis\ttest_cost\t%f" % avg_cost_test)
                print("kpis\ttest_acc\t%f" % accuracy_test)

    train_loop()
if __name__ == '__main__':
    train()
