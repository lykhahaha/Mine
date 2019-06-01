imprt mxnet as mx

class MxSqueezeNet:
    @staticmethod
    def squeeze(data, num_filter):
        # the first part of FIRE module consists of a number of 1x1 filter squuezes
        conv_1x1 = mx.sym.Convolution(data=data, kernel=(1, 1), num_filter=num_filter)
        act_1x1 = mx.sym.LeakyReLU(data=conv_1x1, act_type='elu')

        return act_1x1
    
    @staticmethod
    def fire(data, num_squeeze_filter, num_expand_filter):
        # construct 1x1 squeeze followed by 1x1 expand
        squeeze_1x1 = MxSqueezeNet.squeeze(data, num_squeeze_filter)
        expand_1x1 = mx.sym.Convolution(data=squeeze_1x1, kernel=(1, 1), num_filter=num_expand_filter)
        relu_expand_1x1 = mx.sym.LeakyReLU(data=expand_1x1, act_type='elu')

        # construct 3x3 expand
        exapnd_3x3 = mx.sym.Convolution(data=squeeze_1x1, kernel=(3, 3), pad=(1, 1), num_filter=num_expand_filter)
        relu_expand_3x3 = mx.sym.LeakyReLU(data=exapnd_3x3, act_type='elu')

        # the output is concatenated along channels dimension
        output = mx.sym.Concat(relu_expand_1x1, relu_expand_3x3, dim=1)

        return output

    @staticmethod
    def build(classes):
        # data input
        data = mx.sym.Variable('data')
        
        # Block #1: Conv -> ReLU -> Pool
        conv_1 = mx.sym.Convolution(data=data, kernel=(7, 7), stride=(2, 2), num_filter=96)
        act_1 = mx.sym.LeakyReLU(data=conv_1, act_type='elu')
        pool_1 = mx.sym.Pooling(data=act_1, kernel=(3, 3), pool_type='max', stride=(2, 2))

        # Block #2-4: (FIRE * 3) -> Pool
        fire_2 = MxSqueezeNet.fire(pool_1, num_squeeze_filter=16, num_expand_filter=64)
        fire_3 = MxSqueezeNet.fire(fire_2, num_squeeze_filter=16, num_expand_filter=64)
        fire_4 = MxSqueezeNet.fire(fire_3, num_squeeze_filter=32, num_expand_filter=128)
        pool_4 = mx.sym.Pooling(data=fire_4, kernel=(3, 3), pool_type='max', stride=(2, 2))

        # Block #5-8 : (FIRE) * 4 -> Pool
        fire_5 = MxSqueezeNet.fire(pool_4, num_squeeze_filter=32, num_expand_filter=128)
        fire_6 = MxSqueezeNet.fire(fire_2, num_squeeze_filter=48, num_expand_filter=192)
        fire_7 = MxSqueezeNet.fire(fire_3, num_squeeze_filter=48, num_expand_filter=192)
        fire_8 = MxSqueezeNet.fire(fire_3, num_squeeze_filter=64, num_expand_filter=256)
        pool_8 = mx.sym.Pooling(data=fire_8, kernel=(3, 3), pool_type='max', stride=(2, 2))

        # Last block: FIRE -> Dropout -> Conv -> ACT -> Pool
        fire_9 = MxSqueezeNet.fire(pool_8, num_squeeze_filter=64, num_expand_filter=256)
        do_9 = mx.sym.Dropout(data=fire_9, p=0.5)
        conv_10 = mx.sym.Convolution(data=do_9, kernel=(1, 1), num_filter=classes)
        act_10 = mx.sym.LeakyReLU(data=conv_10, act_type='elu')
        pool_10 = mx.sym.Pooling(data=act_10, kernel=(13, 13), pool_type='avg')

        # softmax classifier
        flatten = mx.sym.Flatten(data=pool_10)
        model = mx.sym.SoftmaxOutput(data=flatten, name='softmax')

        return model