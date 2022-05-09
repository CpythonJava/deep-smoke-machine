# POSTGRADUATE-smoke
工业烟气排放识别的RISE数据集：

[不同类型烟雾视频：高不透明、低不透明、蒸汽、蒸汽与烟](data/dataset/smoke-type.gif)

# <a name="use-this-tool"></a>Use this tool
!!cmd地址位于主文件环境下!!<br/>
1.【使用RISE数据集的JSON文件】(已在data目录下)

2.【将元数据分成三组：训练、验证和测试】<br/>
(将字典中"label_state" and "label_state_admin"聚合到最后的标签中，由新的“label”键表示)
```sh
cd www/init_datasets/
python split_metadata.py confirm
```

3.【将元数据文件中所有视频下载到data/videos/】
```sh
cd www/init_datasets/
python download_videos.py
```

4.【更新optical_flow子模块】
```sh
cd www/optical_flow/
git submodule update --init --recursive
git checkout master
```

5.【处理并保存所有视频为RGB帧到data/rgb/和光流帧到data/flow/】<br/>
【默认只处理RGB帧,如果需要光流帧,在[process_videos.py]将flow_type设为1】
```sh
cd www/init_datasets/
python process_videos.py
```

6.【交叉验证训练模型,已训练的模型会默认存放在data/saved_model/】
```sh
python train.py ssl-i3d-rgb
```
{训练步骤}<br/>
{1.设定初始学习率=0.1}<br/>
{2.保持学习率学习，知道训练误差减小太慢，或者验证误差一直增加}<br/>
{3.降低学习率10倍}<br/>
{4.从之前学习率训练的模型中加载最佳模型权重}<br/>
{5.重复2,3,4直到收敛}

7.【在[TensorBoard]上查看训练和测试结果】
```
cd data/saved_model/
tensorboard --logdir=i3d
```

8.【在测试集中测试模型性能，在混淆矩阵每个单元生成总结视频，TP,TN,FP,FN】
```sh
python test.py ssl-i3d-rgb data/saved_model/i3d/I3D-rgb-s6/model/*****.pt
```

# <a name="code-structure"></a>Code infrastructure
【[base_learner.py](www/bin/base_learner.py)实现fit和test函数，提供共享的功能，比如共享模型加载，模型保存，数据增强和进度日志记录】<br/>
【[i3d_learner.py](www/bin/i3d_learner.py)继承了STCNet模型训练的base_learner.py，提供反向传播和GPU并行计算】<br/>
【[smoke_video_dataset.py](www/bin/smoke_video_dataset.py)数据集定义，用于创建DataLoader类和Dataset类，可以在训练模型时迭代批处理】<br/>
【[opencv_functional.py](www/bin/opencv_functional.py)，用于处理视频帧和视频数据增强】<br/>
【[video_transforms.py](www/bin/video_transforms.py)，用于处理视频帧和视频数据增强】

# <a name="dataset"></a>Dataset
【开放数据集[metadata.json](data/metadata.json)数组中每个元素表示视频的元数据】<br/>
【每个元素都是一个带有键和值的字典，该数据集包含了12567个片段，其中19个不同的视图来自三个监控三个不同工业设施的站点的摄像机。】

【camera_id：摄像机的ID】<br/>
【view_id：相机裁剪的ID，每个视图都是从相机拍摄的全景图中裁剪出来的】<br/>
【id：视频片段的id】<br/>
【label_state：志愿者打的标签】<br/>
【label_state_admin：研究人员打的标签】<br/>
【start_time：对应真实世界，捕捉视频开始的时间】<br/>
【url_root：视频URL根，需要结合url_root + url_part得到完整URL】<br/>
【url_part：视频URL部分，需要结合url_root + url_part得到完整URL】<br/>
【file_name：视频文件名——[camera_id]-[view_id]-[year]-[month]-[day]-[bound_left]-[bound_top]-[bound_right]-[bound_bottom]-[video_height]-[video_width]-[start_frame_number]-[start_epoch_time]-[end_epoch_time]】
