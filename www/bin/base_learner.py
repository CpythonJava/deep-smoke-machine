from abc import ABC, abstractmethod
from dataclasses import replace
from typing import OrderedDict
import torch
from utils.util import check_and_create_dir
import logging
import logging.handlers
import absl.logging
from bin.video_transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomPerspective, RandomErasing, Resize, Normalize, ToTensor
from torchvision.transforms import Compose

# class Reshape(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.shape = shape

#     def forward(self, input):
#         """
#         Reshapes the input according to the shape saved in the view data structure.
#         """
#         batch_size = input.size(0)
#         shape = (batch_size, *self.shape)
#         out = input.reshape(shape)
#         return out

class BaseLearner(ABC):
    def __init__(self, use_cuda=None):
        self.logger = None
        # 判断是否使用CUDA
        if use_cuda is None:
            if torch.cuda.is_available:
                self.use_cuda = True
            else:
                self.use_cuda = False
        else:
            if use_cuda is True and torch.cuda.is_available:
                self.use_cuda = True
            else:
                self.use_cuda = False  

    # 训练模型
    # Output：None
    # 抽象方法表示基类的一个方法，没有实现，所以基类不能被实例化
    @abstractmethod
    def fit(self):
        pass

    # 测试模型
    # Output: None
    @abstractmethod
    def test(self):
        pass

    # 保存模型
    def save(self, model, out_path):
        if model is not None and out_path is not None:
            self.log("Save model weights to " + out_path)
        try:
            # 多GPU线程模型
            state_dict = model.module.state_dict()
        except AttributeError:
            # 单GPU模型
            state_dict = model.state_dict()
        check_and_create_dir(out_path)
        torch.save(state_dict, out_path)

    # 加载模型
    def load(self, model, in_path, ignore_fc=False, fill_dim=False):
        if model is not None and in_path is not None:
            self.log("Load model weights from " + in_path)
            # l_p_model:loaded_pre_model
            # l_s_model:loaded_self_model
            # 加载预训练的数据
            l_p_model = torch.load(in_path)
            # print(l_p_model)
            if "state_dict" in l_p_model:
                l_p_model = l_p_model["state_dict"]
            # 加载自己搭建的网络
            l_s_model = model.state_dict()
            # print(l_s_model)
            # 记录替换的词对
            replace_dict = []
            # 将预训练模型中不存在的网络层添加到replace_dict中
            for key, value in l_p_model.items():
                if key not in l_s_model and key.replace(".net","") in l_s_model:
                    print("Load after remove .net: ", key)
                    replace_dict.append((key, key.replace(".net", "")))
            # 将自训练模型中不存在的网络层添加到replace_dict中
            for key, value in l_s_model.items():
                if key not in l_p_model and key.replace(".net", "") in l_p_model:
                    print("Load after adding .net: ", key)
                    replace_dict.append((key.replace(".net", ""), key))
            # print(replace_dict)
            for key, key_new in replace_dict:
                l_p_model[key_new] = l_p_model.pop(key)
            keys1 = set(list(l_p_model.keys()))
            keys2 = set(list(l_s_model.keys()))
            set_diff = (keys1 - keys2) | (keys2 - keys1)
            # print('### Notice: keys that failed to load:{}'.format(set_diff))
            if ignore_fc:
                print("Ignore fully connected layer weights")
                l_p_model = {key: value for key, value in l_s_model.items() if "fc" not in key}
            # if fill_dim:
                # # 注意这里只适用于Inception-v1 I3D模型
                # print("Auto-fill the mismatched dimension for the i3d model...")
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         if param.data.size() != l_p_model[name].size():
                #             print("\t Found dimension mismatch for:", name)
                #             ds = param.data.size()
                #             ls = l_p_model[name].size()
                #             print("\t\t Desired data size:", param.data.size())
                #             print("\t\t Loaded data size:", l_p_model[name].size())
                #             for i in range(len(ds)):
                #                 diff = ds[i] - ls[i]
                #                 if diff > 0:
                #                     print("\t\t\t Desired dimension %d is larger than the loaded dimension" % i)
                #                     m = l_p_model[name].mean(i).unsqueeze(i)
                #                     print("\t\t\t Compute the missing dimension to have size:", m.size())
                #                     l_p_model[name] = torch.cat([l_p_model[name], m], i)
                #                     print("\t\t\t Loaded data are filled to have size:", l_p_model[name].size())
            # update()函数用于将两个字典合并操作，有相同的就覆盖
            # 将创建的模型使用预训练的模型结构进行更新
            l_s_model.update(l_p_model)
            try:
                model.load_state_dict(l_s_model)
            except:
                self.log("Weights were from nn.DataParallel or DistributedDataParallel...")
                self.log("Remove 'module.' prefix from state_dict keys...")
                new_state_dict = OrderedDict()
                for key, value in l_s_model.items():
                    new_state_dict[key.replace("module.", "")] = value
                model.load_state_dict(new_state_dict)
    
    # 记录信息
    def log(self, msg, lv="i"):
        print(msg)
        if self.logger is not None:
            # 普通信息
            if lv == "i":
                self.logger.info(msg)
            # 警告信息
            elif lv == "w":
                self.logger.warning(msg)
            # 错误信息
            elif lv == "e":
                self.logger.error(msg)
            
    # 创建一个记录器
    def create_logger(self, log_path=None):
        if log_path is None:
            return None
        check_and_create_dir(log_path)
        # 创建RotatingFileHandler类型的handler
        # logging.handlers类的作用是将消息发到handler指定的位置(文件、网络、邮件等)
        # 将日志消息发送到磁盘文件，并支持日志文件按大小切割
        handler = logging.handlers.RotatingFileHandler(log_path, mode="a", maxBytes=100000000, backupCount=200)
        # 消除重复的日志记录
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
        # 配置日志信息的格式
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        # 初始化logger对象
        logger = logging.getLogger(log_path)
        # 设置日志等级
        logger.setLevel(logging.INFO)        
        # 删除旧日志操作的循环步骤
        for old_handler in logger.handlers[:]:
            # 删除旧handler
            logger.removeHandler(old_handler)
        # 更新日志
        logger.addHandler(handler)
        self.logger = logger

    # 数据增强
    def get_transform(self, mode, phase=None, image_size=224):
        if mode == "rgb": # three channels (r, g, b)
            mean = (127.5, 127.5, 127.5)
            std = (127.5, 127.5, 127.5)
        elif mode == "flow": # two channels (x, y)
            mean = (127.5, 127.5)
            std = (127.5, 127.5)
        elif mode == "rgbd": # four channels (r, g, b, dark channel)
            mean = (127.5, 127.5, 127.5, 127.5)
            std = (127.5, 127.5, 127.5, 127.5)
        else:
            return None
        nm = Normalize(mean=mean, std=std) # same as (img/255)*2-1
        tt = ToTensor()
        if phase == "train":
            # 数据增强的不同方式
            # Deals with small camera shifts, zoom changes, and rotations due to wind or maintenance
            # 将给定图像随机裁剪为不同大小和宽高比，然后缩放为指定大小
            rrc = RandomResizedCrop(image_size, scale=(0.9, 1), ratio=(3./4., 4./3.))
            # 随机透视变换
            rp = RandomPerspective(anglex=3, angley=3, anglez=3, shear=3)
            # Improve generalization
            # 以给定的概率随机水平旋转给定的PIL图像
            rhf = RandomHorizontalFlip(p=0.5)
            # Deal with dirts, ants, or spiders on the camera lense
            # 随机擦除
            re = RandomErasing(p=0.5, scale=(0.003, 0.01), ratio=(0.3, 3.3), value=0)
            if mode == "rgb" or mode == "rgbd":
                # Color jitter deals with different lighting and weather conditions
                # 亮度、对比度、饱和度、色调
                cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1), gamma=0.3)
                # 将各个方法按顺序组合
                return Compose([cj, rrc, rp, rhf, tt, nm, re, re])
            elif mode == "flow":
                return Compose([rrc, rp, rhf, tt, nm, re, re])
        else:
            return Compose([Resize(image_size), tt, nm])
