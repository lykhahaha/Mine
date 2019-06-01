import mxnet as mx

class MxGoogLeNet:
    @staticmethod
    def conv_module(data, num_filter, kernel_x, kernel_y, stride=(1, 1), pad=(0, 0), stage='', act_type='relu'):
        conv = mx.sym.Convolution(data=data, kernel=(kernel_x, kernel_y), stride=stride, pad=pad, num_filter=num_filter, name=stage+'_conv')

        assert act_type in ['relu', 'elu', 'prelu'], AssertionError('The activation type does not support')
        act_fn = mx.sym.Activation if act_type == 'relu' else mx.sym.LeakyReLU
        act = act_fn(data=conv, name=stage+'_act')

        bn = mx.sym.BatchNorm(data=conv, name=stage+'_bn')

        return bn
    
    @staticmethod
    def inception_module(data, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, num_1x1_proj, stage=''):
        # first branch of Inception module consiss of 1x1 convolutions
        conv_1x1 = MxGoogLeNet.conv_module(data, num_1x1, 1, 1, stage=stage+'_branch_1')

        # second branch of Inception module consiss of 1x1 convolutions for dimensionality reduction and 3x3 convolutions
        conv_r3x3 = MxGoogLeNet.conv_module(conv_1x1, num_3x3_reduce, 1, 1, stage=stage+'_branch_2_reduce')
        conv_3x3 = MxGoogLeNet.conv_module(conv_r3x3, num_3x3, 3, 3, pad=(1, 1), stage=stage+'_branch_2')

        # third branch of Inception module consiss of 1x1 convolutions for dimensionality reduction and 5x5 convolutions
        conv_r5x5 = MxGoogLeNet.conv_module(conv_3x3, num_5x5_reduce, 1, 1, stage=stage+'_branch_3_reduce')
        conv_5x5 = MxGoogLeNet.conv_module(conv_r5x5, num_5x5, 5, 5, pad=(2, 2), stage=stage+'_branch_3')

        # last branch consists of max pooling and 1x1 convolutions
        pool = mx.sym.Pooling(data=conv_5x5, kernel=(3, 3), pool_type='max', stride=(1, 1), pad=(1, 1), name=stage+'_branch_4_pool')
        conv_proj = MxGoogLeNet.conv_module(pool, num_1x1_proj, 1, 1, stage=stage+'_branch_4')

        # concat along channels axis
        concat = mx.sym.Concat(*[conv_1x1, conv_3x3, conv_5x5, conv_proj])

        return concat

    @staticmethod
    def build(classes):
        # data input
        data = mx.sym.Variable('data')

        # Block #1+2: Conv -> Pool -> Conv -> Conv -> Pool
        conv1_1 = MxGoogLeNet.conv_module(data, 64, 7, 7, stride=(2, 2), pad=(3, 3), stage='block_1')
        pool_1 = mx.sym.Pooling(data=conv1_1, kernel=(3, 3), pool_type='max', stride=(2, 2), pad=(1, 1), name='pool_1')
        conv2_1 = MxGoogLeNet.conv_module(pool_1, 64, 1, 1, stage='block_2_reduce')
        conv2_2 = MxGoogLeNet.conv_module(conv2_1, 192, 3, 3, pad=(1, 1), stage='block_2')
        pool_2 = mx.sym.Pooling(data=conv2_2, kernel=(3, 3), pool_type='max', stride=(2, 2), pad=(1, 1), name='pool_2')

        # Block #3: 2 inception modules are stacked onn top -> Max Pool
        in_3a = MxGoogLeNet.inception_module(pool_2, 64, 96, 128, 16, 32, 32, stage='block_3a')
        in_3b = MxGoogLeNet.inception_module(in_3a, 128, 128, 192, 32, 96, 64, stage='block_3b')
        pool_3 = mx.sym.Pooling(data=in_3b, kernel=(3, 3), pool_type='max', stride=(2, 2), pad=(1, 1), name='pool_3')

        # Block #4: 4 inception modules are stacked onn top -> Max Pool
        in_4a = MxGoogLeNet.inception_module(pool_3, 192, 96, 208, 16, 48, 64, stage='block_4a')
        in_4b = MxGoogLeNet.inception_module(in_4a, 160, 112, 224, 24, 64, 64, stage='block_4b')
        in_4c = MxGoogLeNet.inception_module(in_4b, 128, 128, 256, 24, 64, 64, stage='block_4c')
        in_4d = MxGoogLeNet.inception_module(in_4c, 112, 144, 288, 32, 64, 64, stage='block_4d')
        in_4e = MxGoogLeNet.inception_module(in_4d, 256, 160, 320, 32, 128, 128, stage='block_4e')
        pool_4 = mx.sym.Pooling(data=in_4e, kernel=(3, 3), pool_type='max', stride=(2, 2), pad=(1, 1), name='pool_4')

        # Block #5: 2 inception modules are stacked onn top -> avg pool -> dropout -> FC -> Softmax
        in_5a = MxGoogLeNet.inception_module(pool_4, 256, 160, 320, 32, 128, 128, stage='block_5a')
        in_5b = MxGoogLeNet.inception_module(in_5a, 384, 192, 384, 48, 128, 128, stage='block_5b')
        pool_5 = mx.sym.Pooling(data=in_5b, kernel=(7, 7), pool_type='avg', stride=(1, 1), name='pool_5')
        do = mx.sym.Dropout(data=pool_5, p=0.5)
        flatten = mx.sym.Flatten(data=do)
        fc_1 = mx.sym.FullyConnected(data=flatten, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fc_1, name='softmax')

        return model