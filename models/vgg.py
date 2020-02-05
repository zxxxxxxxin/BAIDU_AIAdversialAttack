from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid
__all__ = ["VGG16"]
class VGG():
    def __init__(self):
        self.name = 'vgg'
    def x2paddle_net(self,input):
        #_input_1 = fluid.layers.data(name='_input_1', dtype='float32', shape=[1, 3, 224, 224], append_batch_size=False)
        #_input_1 = fluid.layers.create_parameter(name="_input_1",
        #                                         shape=(1,3,224,224),
        #                                         dtype='float32',
        #                                         )
        _input_1 = input
        _classifier_0_bias = fluid.layers.create_parameter(default_initializer=Constant(0.0), name='_classifier_0_bias', dtype='float32', shape=[4096], attr='_classifier_0_bias')
        _classifier_0_weight = fluid.layers.create_parameter(default_initializer=Constant(0.0), name='_classifier_0_weight', dtype='float32', shape=[4096, 25088], attr='_classifier_0_weight')
        _classifier_3_bias = fluid.layers.create_parameter(default_initializer=Constant(0.0), name='_classifier_3_bias', dtype='float32', shape=[4096], attr='_classifier_3_bias')
        _classifier_3_weight = fluid.layers.create_parameter(default_initializer=Constant(0.0), name='_classifier_3_weight', dtype='float32', shape=[4096, 4096], attr='_classifier_3_weight')
        _classifier_6_bias = fluid.layers.create_parameter(default_initializer=Constant(0.0), name='_classifier_6_bias', dtype='float32', shape=[121], attr='_classifier_6_bias')
        _classifier_6_weight = fluid.layers.create_parameter(default_initializer=Constant(0.0), name='_classifier_6_weight', dtype='float32', shape=[121, 4096], attr='_classifier_6_weight')
        _33 = fluid.layers.conv2d(_input_1, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_33', bias_attr='_features_0_bias', groups=1, num_filters=64, param_attr='_features_0_weight')
        _34 = fluid.layers.relu(_33, name='_34')
        _35 = fluid.layers.conv2d(_34, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_35', bias_attr='_features_2_bias', groups=1, num_filters=64, param_attr='_features_2_weight')
        _36 = fluid.layers.relu(_35, name='_36')
        _37 = fluid.layers.pool2d(_36, ceil_mode=False, pool_padding=[0, 0], exclusive=False, name='_37', pool_size=[2, 2], pool_stride=[2, 2], pool_type='max')
        _38 = fluid.layers.conv2d(_37, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_38', bias_attr='_features_5_bias', groups=1, num_filters=128, param_attr='_features_5_weight')
        _39 = fluid.layers.relu(_38, name='_39')
        _40 = fluid.layers.conv2d(_39, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_40', bias_attr='_features_7_bias', groups=1, num_filters=128, param_attr='_features_7_weight')
        _41 = fluid.layers.relu(_40, name='_41')
        _42 = fluid.layers.pool2d(_41, ceil_mode=False, pool_padding=[0, 0], exclusive=False, name='_42', pool_size=[2, 2], pool_stride=[2, 2], pool_type='max')
        _43 = fluid.layers.conv2d(_42, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_43', bias_attr='_features_10_bias', groups=1, num_filters=256, param_attr='_features_10_weight')
        _44 = fluid.layers.relu(_43, name='_44')
        _45 = fluid.layers.conv2d(_44, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_45', bias_attr='_features_12_bias', groups=1, num_filters=256, param_attr='_features_12_weight')
        _46 = fluid.layers.relu(_45, name='_46')
        _47 = fluid.layers.conv2d(_46, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_47', bias_attr='_features_14_bias', groups=1, num_filters=256, param_attr='_features_14_weight')
        _48 = fluid.layers.relu(_47, name='_48')
        _49 = fluid.layers.pool2d(_48, ceil_mode=False, pool_padding=[0, 0], exclusive=False, name='_49', pool_size=[2, 2], pool_stride=[2, 2], pool_type='max')
        _50 = fluid.layers.conv2d(_49, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_50', bias_attr='_features_17_bias', groups=1, num_filters=512, param_attr='_features_17_weight')
        _51 = fluid.layers.relu(_50, name='_51')
        _52 = fluid.layers.conv2d(_51, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_52', bias_attr='_features_19_bias', groups=1, num_filters=512, param_attr='_features_19_weight')
        _53 = fluid.layers.relu(_52, name='_53')
        _54 = fluid.layers.conv2d(_53, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_54', bias_attr='_features_21_bias', groups=1, num_filters=512, param_attr='_features_21_weight')
        _55 = fluid.layers.relu(_54, name='_55')
        _56 = fluid.layers.pool2d(_55, ceil_mode=False, pool_padding=[0, 0], exclusive=False, name='_56', pool_size=[2, 2], pool_stride=[2, 2], pool_type='max')
        _57 = fluid.layers.conv2d(_56, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_57', bias_attr='_features_24_bias', groups=1, num_filters=512, param_attr='_features_24_weight')
        _58 = fluid.layers.relu(_57, name='_58')
        _59 = fluid.layers.conv2d(_58, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_59', bias_attr='_features_26_bias', groups=1, num_filters=512, param_attr='_features_26_weight')
        _60 = fluid.layers.relu(_59, name='_60')
        _61 = fluid.layers.conv2d(_60, dilation=[1, 1], filter_size=[3, 3], stride=[1, 1], padding=[1, 1], name='_61', bias_attr='_features_28_bias', groups=1, num_filters=512, param_attr='_features_28_weight')
        _62 = fluid.layers.relu(_61, name='_62')
        _63 = fluid.layers.pool2d(_62, ceil_mode=False, pool_padding=[0, 0], exclusive=False, name='_63', pool_size=[2, 2], pool_stride=[2, 2], pool_type='max')
        _64 = fluid.layers.pool2d(_63, ceil_mode=False, pool_padding=[0, 0], name='_64', exclusive=True, pool_size=[1, 1], pool_stride=[1, 1], pool_type='avg')
        _65 = fluid.layers.flatten(_64, axis=1, name='_65')
        _66_mm = fluid.layers.matmul(y=_classifier_0_weight, x=_65, name='_66_mm', transpose_x=False, alpha=1.0, transpose_y=True)
        _66 = fluid.layers.elementwise_add(y=_classifier_0_bias, x=_66_mm, name='_66')
        _67 = fluid.layers.relu(_66, name='_67')
        _68_mm = fluid.layers.matmul(y=_classifier_3_weight, x=_67, name='_68_mm', transpose_x=False, alpha=1.0, transpose_y=True)
        _68 = fluid.layers.elementwise_add(y=_classifier_3_bias, x=_68_mm, name='_68')
        _69 = fluid.layers.relu(_68, name='_69')
        _70_mm = fluid.layers.matmul(y=_classifier_6_weight, x=_69, name='_70_mm', transpose_x=False, alpha=1.0, transpose_y=True)
        _70 = fluid.layers.elementwise_add(y=_classifier_6_bias, x=_70_mm, name='_70')

        return [_input_1], [_70]

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

def VGG16():
    model = VGG()
    return model
    