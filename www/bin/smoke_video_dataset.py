import os
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.util import *

# 视频烟雾数据集
class SmokeVideoDataset(Dataset):
    def __init__(self, metadata_path=None, root_dir=None, transform=None):
        '''
        metapath:video metadata json文件的地址
        root_dir:存放视频文件的根目录
        transform：在视频上应用可选的变换
        '''
        self.metadata = load_json(metadata_path)
        self.root_dir = root_dir
        self.transform = transform
    
    # 使用：len(p)
    def __len__(self):
        return len(self.metadata)
    
    # 使用：p[key] 
    def __getitem__(self, idx):
        v = self.metadata[idx]

        # 加载视频数据
        # 文件地址
        file_path = os.path.join(self.root_dir, v["file_name"] + ".npy")
        # 判断文件地址是否有效
        if not is_file_here(file_path):
            raise ValueError("Cannot find file: %s" % (file_path))
        
        # 加载帧,将格式改成uint8
        frames = np.load(file_path).astype(np.uint8)
        t = frames.shape[0]

        # 转换视频
        if self.transform:
            frames = self.transform(frames)

        # 加载标签
        label = v["label"]
        if label == 1:
            labels = np.array([0.0, 1.0], dtype=np.float32) # 第2列表示是的概率
        else:
            labels = np.array([0.0, 1.0], dtype=np.float32) # 第1列表示不是的概率
        # 每帧重复（逐帧检测）
        labels = np.repeat([labels], t, axis=0)

        return {"frames":frames,
                "labels":self.labels_to_tensor(labels),
                "file_name":v["file_name"]}
    
    def labels_to_tensor(self, labels):
        '''
        将numpy.ndarray的格式(time, x, num_of_action_classes)转换成
        (num_of_action_classes, x, time)
        '''
        return torch.from_numpy(labels.transpose([1,0]))  



    