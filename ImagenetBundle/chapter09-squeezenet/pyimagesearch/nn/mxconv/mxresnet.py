import mxnet as mx

class MxResnet:
    @staticmethod
    def residual_module(data, num_filter, stride, red=False, bn_eps=2e-5, bn_mom=0.9):
        # define shortcut which is used for identity mapping
        shortcut = data

        # first block of the ResNet module are the 1x1 Convs
        bn_1 = mx.sym.BatchNorm(data=data, eps=bn_eps, momentum=bn_mom, fix_gamma=False)
        act_1 = mx.sym.Activation(data=bn_1, act_type='relu')
        conv_1 = mx.sym.Convolution(data=act_1, kernel=(1, 1), stride=(1, 1), num_filter=num_filter//4, no_bias=True)

        # second block of Resnet module are 3x3 Convs
        bn_2 = mx.sym.BatchNorm(data=conv_1, eps=bn_eps, momentum=bn_mom, fix_gamma=False)
        act_2 = mx.sym.Activation(data=bn_2, act_type='relu')
        conv_2 = mx.sym.Convolution(data=act_2, kernel=(3, 3), stride=stride, pad=(1, 1), num_filter=num_filter//4, no_bias=True)

        # third block of Resnet module are 1x1 Convs
        bn_3 = mx.sym.BatchNorm(data=conv_2, eps=bn_eps, momentum=bn_mom, fix_gamma=False)
        act_3 = mx.sym.Activation(data=bn_3, act_type='relu')
        conv_3 = mx.sym.Convolution(data=act_3, kernel=(1, 1), stride=(1, 1), num_filter=num_filter, no_bias=True)

        # if reducing spatial size, reduce dims of activation of original data by conv
        if red:
            shortcut = mx.sym.Convolution(data=act_1, kernel=(1, 1), stride=stride, num_filter=num_filter, no_bias=True)
        
        # add shortcut and final conv
        add = shortcut + conv_3

        return add

    @staticmethod
    def build(classes, stages, filters, bn_eps=2e-5, bn_mom=0.9):
        # data input
        data = mx.sym.Variable('data')

        # Block #1: BN -> Conv -> BN -> ReLU -> Pooling
        bn1_1 = mx.sym.BatchNorm(data=data, eps=bn_eps, momentum=bn_mom, fix_gamma=False)
        conv1_1 = mx.sym.Convolution(data=bn1_1, kernel=(7, 7), stride=(2, 2), pad=(3, 3), num_filter=filters[0], no_bias=True)
        bn1_2 = mx.sym.BatchNorm(data=conv1_1, eps=bn_eps, momentum=bn_mom, fix_gamma=False)
        act1_2 = mx.sym.Activation(data=bn1_2, act_type='relu')
        pool_1 = mx.sym.Pooling(data=act1_2, kernel=(3, 3), pool_type='max', stride=(2, 2), pad=(1, 1), fix_gamma=False)
        body = pool_1
        
        # loop over the stages
        for i, stage in enumerate(stages):
            # initialize the stride, then apply a residual module used to reduce spatial size of input volume
            stride = (1, 1) if i == 0 else (2, 2)
            body = MxResnet.residual_module(body, filters[i+1], stride, red=True, bn_eps=bn_eps, bn_mom=bn_mom)

            for _ in range(stage - 1):
                # apply ResNet module
                body = MxResnet.residual_module(body, filters[i+1], (1, 1), bn_eps=bn_eps, bn_mom=bn_mom)

        # apply BN -> ReLU -> Pool
        bn2_1 = mx.sym.BatchNorm(data=body, eps=bn_eps, momentum=bn_mom, fix_gamma=False)
        act2_1 = mx.sym.Activation(data=bn2_1, act_type='relu')
        pool_2 = mx.sym.Pooling(data=act2_1, kernel= (7, 7), pool_type='avg')

        # softmax classifier
        flatten = mx.sym.Flatten(data=pool_2)
        fc_1 = mx.sym.FullyConnected(data=flatten, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fc_1, name='softmax')

        return model