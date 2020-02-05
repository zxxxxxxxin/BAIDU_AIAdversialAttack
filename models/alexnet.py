from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid
__all__ = ["alexnet"]
class alex():
    def __init__(self):
        self.name = 'alexnet'
    def x2paddle_net(self,input):
        #_input_1 = fluid.layers.data(name='_input_1', dtype='float32', shape=[1, 3, 224, 224], append_batch_size=False)
        #_input_1 = fluid.layers.create_parameter(name="_input_1",
        #                                         shape=(1,3,224,224),
        #                                         dtype='float32',
        #                                         )
        _input_1 = input
        #_input_1 = fluid.layers.data(append_batch_size=False, dtype='float32', name='_input_1', shape=[1, 3, 224, 224])
        classifier_1_bias = fluid.layers.create_parameter(attr='classifier_1_bias', default_initializer=Constant(0.0), dtype='float32', name='classifier_1_bias', shape=[4096])
        classifier_1_weight = fluid.layers.create_parameter(attr='classifier_1_weight', default_initializer=Constant(0.0), dtype='float32', name='classifier_1_weight', shape=[4096, 9216])
        classifier_4_bias = fluid.layers.create_parameter(attr='classifier_4_bias', default_initializer=Constant(0.0), dtype='float32', name='classifier_4_bias', shape=[4096])
        classifier_4_weight = fluid.layers.create_parameter(attr='classifier_4_weight', default_initializer=Constant(0.0), dtype='float32', name='classifier_4_weight', shape=[4096, 4096])
        classifier_6_bias = fluid.layers.create_parameter(attr='classifier_6_bias', default_initializer=Constant(0.0), dtype='float32', name='classifier_6_bias', shape=[121])
        classifier_6_weight = fluid.layers.create_parameter(attr='classifier_6_weight', default_initializer=Constant(0.0), dtype='float32', name='classifier_6_weight', shape=[121, 4096])
        _17 = fluid.layers.conv2d(_input_1, filter_size=[11, 11], param_attr='features_0_weight', dilation=[1, 1], groups=1, num_filters=64, padding=[2, 2], stride=[4, 4], name='_17', bias_attr='features_0_bias')
        _18 = fluid.layers.relu(_17, name='_18')
        _19 = fluid.layers.pool2d(_18, exclusive=False, pool_stride=[2, 2], pool_type='max', pool_size=[3, 3], pool_padding=[0, 0], ceil_mode=False, name='_19')
        _20 = fluid.layers.conv2d(_19, filter_size=[5, 5], param_attr='features_3_weight', dilation=[1, 1], groups=1, num_filters=192, padding=[2, 2], stride=[1, 1], name='_20', bias_attr='features_3_bias')
        _21 = fluid.layers.relu(_20, name='_21')
        _22 = fluid.layers.pool2d(_21, exclusive=False, pool_stride=[2, 2], pool_type='max', pool_size=[3, 3], pool_padding=[0, 0], ceil_mode=False, name='_22')
        _23 = fluid.layers.conv2d(_22, filter_size=[3, 3], param_attr='features_6_weight', dilation=[1, 1], groups=1, num_filters=384, padding=[1, 1], stride=[1, 1], name='_23', bias_attr='features_6_bias')
        _24 = fluid.layers.relu(_23, name='_24')
        _25 = fluid.layers.conv2d(_24, filter_size=[3, 3], param_attr='features_8_weight', dilation=[1, 1], groups=1, num_filters=256, padding=[1, 1], stride=[1, 1], name='_25', bias_attr='features_8_bias')
        _26 = fluid.layers.relu(_25, name='_26')
        _27 = fluid.layers.conv2d(_26, filter_size=[3, 3], param_attr='features_10_weight', dilation=[1, 1], groups=1, num_filters=256, padding=[1, 1], stride=[1, 1], name='_27', bias_attr='features_10_bias')
        _28 = fluid.layers.relu(_27, name='_28')
        _29 = fluid.layers.pool2d(_28, exclusive=False, pool_stride=[2, 2], pool_type='max', pool_size=[3, 3], pool_padding=[0, 0], ceil_mode=False, name='_29')
        _30 = fluid.layers.pool2d(_29, exclusive=True, pool_stride=[1, 1], pool_type='avg', pool_size=[1, 1], pool_padding=[0, 0], ceil_mode=False, name='_30')
        _31 = fluid.layers.flatten(_30, name='_31', axis=1)
        _32_mm = fluid.layers.matmul(x=_31, y=classifier_1_weight, transpose_x=False, transpose_y=True, name='_32_mm', alpha=1.0)
        _32 = fluid.layers.elementwise_add(x=_32_mm, y=classifier_1_bias, name='_32')
        _33 = fluid.layers.relu(_32, name='_33')
        _34_mm = fluid.layers.matmul(x=_33, y=classifier_4_weight, transpose_x=False, transpose_y=True, name='_34_mm', alpha=1.0)
        _34 = fluid.layers.elementwise_add(x=_34_mm, y=classifier_4_bias, name='_34')
        _35 = fluid.layers.relu(_34, name='_35')
        _36_mm = fluid.layers.matmul(x=_35, y=classifier_6_weight, transpose_x=False, transpose_y=True, name='_36_mm', alpha=1.0)
        _36 = fluid.layers.elementwise_add(x=_36_mm, y=classifier_6_bias, name='_36')
        
        return [_input_1], [_36]

def run_net(param_dir="./"):
    import os
    inputs, outputs = x2paddle_net()
    for i, out in enumerate(outputs):
        if isinstance(out, list):
            for out_part in out:
                outputs.append(out_part)
            del outputs[i]
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        b = os.path.exists(os.path.join(param_dir, var.name))
        return b

    fluid.io.load_vars(exe,
                       param_dir,
                       fluid.default_main_program(),
                       predicate=if_exist)

def alexnet():
    model = alex()
    return model
    