from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 3D最大池化
class MaxPool3dSamePadding(nn.MaxPool3d):

    # 计算各个维度的padding
    def compute_pad(self, dimension, s_channel):
        if s_channel % self.stride[dimension] == 0:
            return max(self.kernel_size[dimension] - self.stride[dimension], 0)
        else:
            return max(self.kernel_size[dimension] - (s_channel % self.stride[dimension]), 0)    

    def forward(self, x):
        # 计算不同维度的padding
        (batch, channel, time, height, width) = x.size()
        # 计算大于等于该值的最小整数
        out_time = np.ceil(float(time) / float(self.stride[0]))
        out_height = np.ceil(float(height) / float(self.stride[1]))
        out_width = np.ceil(float(width) / float(self.stride[2]))
        
        pad_time = self.compute_pad(0, time)
        pad_height = self.compute_pad(1, height)
        pad_width = self.compute_pad(2, width)

        # 将pad_time，pad_height，pad_width平分后扩充到相应维度
        # 令pad_** = pad_**_f + pad_**_b
        pad_time_f = pad_time // 2
        pad_time_b = pad_time - pad_time_f
        pad_height_f = pad_height // 2
        pad_height_b = pad_height - pad_height_f
        pad_width_f = pad_width // 2
        pad_width_b = pad_width - pad_width_f

        # 将所有需要扩充的维度合并为pad
        pad = (pad_width_f, pad_width_b, pad_height_f, pad_height_b, pad_time_f, pad_time_b)
        x = F.pad(x, pad)
        # 返回父类的方法
        return super(MaxPool3dSamePadding, self).forward(x)


# 3D单元
class Unit3D(nn.Module):

    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0,
            activation_fn=F.relu,
            use_batch_norm=True,
            use_bias=False,
            name="unit_3d"
            ):
        
        "初始化Unit3D模型"
        super(Unit3D, self).__init__()

        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self.padding = padding
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self.name = name

        # 搭建3D CNN
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._out_channels,
                                kernel_size=self._kernel_size,
                                stride=self._stride,
                                padding=0,# 我们将根据输入大小在forward函数中动态pad
                                bias=self._use_bias
                                )
        # 是否使用BN
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._out_channels, eps=0.001, momentum=0.01)

    # 计算各个维度的padding
    def compute_pad(self, dimension, s_channel):
        if s_channel % self._stride[dimension] == 0:
            return max(self._kernel_size[dimension] - self._stride[dimension], 0)
        else:
            return max(self._kernel_size[dimension] - (s_channel % self._stride[dimension]), 0)    

    def forward(self, x):
        # 计算不同维度的padding
        (batch, channel, time, height, width) = x.size()
        # 计算大于等于该值的最小整数
        out_time = np.ceil(float(time) / float(self._stride[0]))
        out_height = np.ceil(float(height) / float(self._stride[1]))
        out_width = np.ceil(float(width) / float(self._stride[2]))
        
        pad_time = self.compute_pad(0, time)
        pad_height = self.compute_pad(1, height)
        pad_width = self.compute_pad(2, width)

        # 将pad_time，pad_height，pad_width平分后扩充到相应维度
        # 令pad_** = pad_**_f + pad_**_b
        pad_time_f = pad_time // 2
        pad_time_b = pad_time - pad_time_f
        pad_height_f = pad_height // 2
        pad_height_b = pad_height - pad_height_f
        pad_width_f = pad_width // 2
        pad_width_b = pad_width - pad_width_f

        # 将所有需要扩充的维度合并为pad
        pad = (pad_width_f, pad_width_b, pad_height_f, pad_height_b, pad_time_f, pad_time_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        # 是否加入BN
        if self._use_batch_norm:
            x = self.bn(x)
        # 是否加入激活函数
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x
        

# 元模块
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        # branch0结构
        self.b0 = Unit3D(in_channels=in_channels, out_channels=out_channels[0],
                        kernel_size=[1, 1, 1], padding=0,
                        name=name+'/Branch_0/Conv3d_0a_1x1'
                        )
        # branch1结构
        self.b1a = Unit3D(in_channels=in_channels, out_channels=out_channels[1],
                        kernel_size=[1, 1, 1], padding=0,
                        name=name+'/Branch_1/Conv3d_0a_1x1'
                        )
        self.b1b = Unit3D(in_channels=out_channels[1], out_channels=out_channels[2],
                        kernel_size=[3, 3, 3],
                        name=name+'/Branch_1/Conv3d_0b_3x3'
                        )
        # branch2结构
        self.b2a = Unit3D(in_channels=in_channels, out_channels=out_channels[3],
                        kernel_size=[1, 1, 1], padding=0,
                        name=name+'/Branch_2/Conv3d_0a_1x1'
                        )
        self.b2b = Unit3D(in_channels=out_channels[3], out_channels=out_channels[4],
                        kernel_size=[3, 3, 3],
                        name=name+'/Branch_2/Conv3d_0b_3x3'
                        )
        # branch3结构
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, out_channels=out_channels[5],
                        kernel_size=[1, 1, 1], padding=0,
                        name=name+'/Branch_3/Conv3d_0b_1x1'
                        )    
        self.name = name 

    def forward(self, x):
        branch0 = self.b0(x)  
        branch1 = self.b1b(self.b1a(x)) 
        branch2 = self.b2b(self.b2a(x))   
        branch3 = self.b3b(self.b3a(x))    
        return torch.cat([branch0,branch1,branch2,branch3], dim=1)     


# I3D元模块
class InceptionI3d(nn.Module):
    '''Inception-v1 I3D结构'''

    # 按顺序排列模型的端点
    # 过程中，所有端点在达到'final_endpoint'后都在字典中作为第二个返回值返回。
    # 结构列表
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',    #1
        'MaxPool3d_2a_3x3', #2
        'Conv3d_2b_1x1',    
        'Conv3d_2c_3x3',    
        'MaxPool3d_3a_3x3', #3
        'Mixed_3b',         
        'Mixed_3c',         
        'MaxPool3d_4a_3x3', #4
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2', #5
        'Mixed_5b',
        'Mixed_5c',
        'Logits',           # full—linear
        'Predecitions',     # prediction
    )
    def __init__(self, num_classes=400, spatial_squeeze=True,
                final_endpoint='Logits', name='inception_i3d',
                in_channels=3, dropout_keep_prob=0.5
                ):
        '''初始化I3D模型实例：
        Args：
            num_classes:输出层的输出分类数
            spatial_squeeze:是否在返回之前压缩logit的空间维度
            final_endpoint：指定要构建的模型的最后一个端点，final_endpoint之前所有端点输出也会在return时返回
            name:该模块的名字
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        '''

        # 如果final_endpoint没有出现在列表中
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        # 初始化
        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)
        
        # 构建网络蓝图
        self.end_points = {}

        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, 
                                            out_channels=64,
                                            kernel_size=[7, 7, 7],
                                            stride=(2, 2, 2),
                                            padding=(3, 3, 3),
                                            name=name+end_point)
        if self._final_endpoint == end_point:return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
                                            kernel_size=[1, 3, 3],
                                            stride=(1, 2, 2),
                                            padding=0)
        if self._final_endpoint == end_point:return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, 
                                            out_channels=64,
                                            kernel_size=[1, 1, 1],
                                            padding=0,
                                            name=name+end_point)
        if self._final_endpoint == end_point:return
        
        end_point = 'Conv3d_2c_3x3' 
        self.end_points[end_point] = Unit3D(in_channels=64, 
                                            out_channels=192,
                                            kernel_size=[3, 3, 3],
                                            padding=1,
                                            name=name+end_point)  
        if self._final_endpoint == end_point:return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
                                            kernel_size=[1, 3, 3],
                                            stride=(1, 2, 2),
                                            padding=0)   
        if self._final_endpoint == end_point:return

        end_point = 'Mixed_3b'   
        self.end_points[end_point] = InceptionModule(192,[64,96,128,16,32,32], name+end_point)   
        if self._final_endpoint == end_point:return 
        # in_channels = 上一级Inc的out_channels中[0]+[2]+[4]+[5]
        end_point = 'Mixed_3c'  
        self.end_points[end_point] = InceptionModule(64+128+32+32,[128,128,192,32,96,64], name+end_point)   
        if self._final_endpoint == end_point:return  
             
        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
                                            kernel_size=[3, 3, 3],
                                            stride=(2, 2, 2),
                                            padding=0)   
        if self._final_endpoint == end_point:return
        
        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64,[192,96,208,16,48,64], name+end_point)  
        if self._final_endpoint == end_point:return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64,[160,112,224,24,64,64], name+end_point)  
        if self._final_endpoint == end_point:return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64,[128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point:return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64,[112,144,288,32,64,64], name+end_point)   
        if self._final_endpoint == end_point:return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64,[256,160,320,32,128,128], name+end_point)   
        if self._final_endpoint == end_point:return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                            padding=0)  
        if self._final_endpoint == end_point:return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, out_channels=self._num_classes,
                            kernel_size=[1, 1, 1], padding=0, activation_fn=None,
                            use_batch_norm=False, use_bias=True, name='logits')

        # 搭建指令
        self.build()

    # 将模型结构依次添加创建网络结构
    def build(self):
        for key in self.end_points.keys():
            self.add_module(key, self.end_points[key])

    def forward(self, x, no_logits=False):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                # 使用modules
                x = self._modules[end_point](x)
        
        # Logits作为最后一层
        if self._final_endpoint == 'Logits' and no_logits is False:
            x = self.logits(self.dropout(self.avg_pool(x)))
            # 将空间维度压缩
            if self._spatial_squeeze:
                x = x.squeeze(3).squeeze(3)
                # logits输出为(batch, classes, time)
        return x

    # 更替最后的logits层 
    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, out_channels=self._num_classes,
                             kernel_size=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    # def extract_features(self, x):
    #     for end_point in self.VALID_ENDPOINTS:
    #         if end_point in self.end_points:
    #             x = self._modules[end_point](x)
    #     return self.avg_pool(x)

    # def extract_conv_output(self, x):
    #     for end_point in self.VALID_ENDPOINTS:
    #         if end_point in self.end_points:
    #             x = self._modules[end_point](x)
    #     return x

    # def conv_output_to_model_output(self, x):
    #     x = self.logits(self.dropout(self.avg_pool(x)))
    #     if self._spatial_squeeze:
    #         logits = x.squeeze(3).squeeze(3)
    #     # logits is batch X time X classes, which is what we want to work with
    #     return logits