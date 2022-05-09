import os
from model.pytorch_i3d import InceptionI3d
from bin.smoke_video_dataset import SmokeVideoDataset
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # 使用nvidia-smi命令中的顺序
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # 指明使用哪些GPU
from bin.base_learner import BaseLearner

from utils.util import *
import shutil
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report as cr
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score
import torch.distributed as dist
import re

# I3D模型
class I3d(BaseLearner):
    def __init__(self,
            use_cuda=None,      # 是否使用cuda
            use_tsm=False,      # 是否使用TSM(Temporal Shift model)
            use_nl=False,       # 是否使用Non-local model
            use_tc=False,       # 是否使用Timeception model
            use_lstm=False,     # 是否使用LSTM model
            freeze_i3d=False,   # 当训练Timeception时是否冻住i3d层
            batch_size_train=5, # 训练时每个batch_size
            batch_size_test=5,  # 测试时每个batch_size
            batch_size_extract_features = 40, # 提取特征时每个batch_size
            max_steps=12000,    # 总共训练步数
            num_steps_per_update=2, # 梯度积累
            init_lr=0.01,       # 初始化学习率
            weight_decay=0.000001, # L2正则化
            momentum=0.9,       # SGD参数
            milestones=[500,1500],# MultiStepLR参数
            gamma=0.1,          # MultiStepLR参数
            num_classes=2,      # 分类种类
            num_steps_per_check=50,# 每经过多少步保存模型并记录日志
            parallel=True,      # 是否使用nn.DistributeDataParallel
            augment=True,       # 是否使用数据增强
            num_workers=1,    # 数据加载使用的线程数量
            mode="rgb",         # 使用的方式
            path_frame="../data/rgb/", # 加载视频帧的地址
            code_testing=False  # 代码奏效的Flag标志位
            ):
        super().__init__(use_cuda=use_cuda)

        # 初始化对象属性
        self.use_cuda = use_cuda     
        self.use_tsm = use_tsm
        self.use_nl = use_nl
        self.use_tc = use_tc
        self.use_lstm = use_lstm
        self.freeze_i3d = freeze_i3d
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.batch_size_extract_features = batch_size_extract_features
        self.max_steps = max_steps
        self.num_steps_per_update = num_steps_per_update
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.milestones = milestones
        self.gamma = gamma
        self.num_classes = num_classes
        self.num_steps_per_check = num_steps_per_check
        self.parallel = parallel
        self.augment = augment
        self.num_workers = num_workers
        self.mode = mode
        self.path_frame = path_frame
        self.code_testing = code_testing

        # 内部参数
        # 模型输入尺寸
        self.image_size = 224
        # 是否可以并行
        self.can_parallel = False

        # 测试模型
        self.code_testing = code_testing
        if code_testing:
                self.max_steps = 10
   
    # 记录参数信息
    def log_parameters(self):
        text = "\nParameters:\n"
        text += "use_tsm: " + str(self.use_tsm) + "\n"
        text += "  use_nl: " + str(self.use_nl) + "\n"
        text += "  use_tc: " + str(self.use_tc) + "\n"
        text += "  use_lstm: " + str(self.use_lstm) + "\n"
        text += "  freeze_i3d: " + str(self.freeze_i3d) + "\n"
        text += "  batch_size_train: " + str(self.batch_size_train) + "\n"
        text += "  batch_size_test: " + str(self.batch_size_test) + "\n"
        text += "  batch_size_extract_features: " + str(self.batch_size_extract_features) + "\n"
        text += "  max_steps: " + str(self.max_steps) + "\n"
        text += "  num_steps_per_update: " + str(self.num_steps_per_update) + "\n"
        text += "  init_lr: " + str(self.init_lr) + "\n"
        text += "  weight_decay: " + str(self.weight_decay) + "\n"
        text += "  momentum: " + str(self.momentum) + "\n"
        text += "  milestones: " + str(self.milestones) + "\n"
        text += "  gamma: " + str(self.gamma) + "\n"
        text += "  num_classes: " + str(self.num_classes) + "\n"
        text += "  num_steps_per_check: " + str(self.num_steps_per_check) + "\n"
        text += "  parallel: " + str(self.parallel) + "\n"
        text += "  augment: " + str(self.augment) + "\n"
        text += "  num_workers: " + str(self.num_workers) + "\n"
        text += "  mode: " + self.mode + "\n"
        text += "  path_frame: " + self.path_frame + "\n"
        self.log(text)

    # 设置模型
    def set_model(self, rank, world_size, mode, path_model, parallel, phase="train"):
        # 根据训练或测试选择batch_size
        if phase == "train":
            model_batch_size = self.batch_size_train
        elif phase == "test":
            model_batch_size = self.batch_size_test
        elif phase == "feature":
            model_batch_size = self.batch_size_extract_features
        
        # 基于mode安装模型
        has_extra_layers = self.use_tc or self.use_tsm or self.use_nl or self.use_lstm
        # 因为使用Kinetics数据集预训练的网络，所以需要400类别数
        num_kinetics = 400
        if mode == "rgb" or mode == "rgbd":
            in_channels = 3 if mode == "rgb" else 4
            if not has_extra_layers:
                model = InceptionI3d(num_classes=num_kinetics, in_channels=in_channels)
            else:
                # 输入尺寸(batch_size, channel, time, height, width)
                input_size = [model_batch_size, in_channels, 36, 224, 224]
        elif mode == 'flow':
            in_channels = 2
            # 单纯使用I3D模型
            if not has_extra_layers:
                model = InceptionI3d(num_classes=num_kinetics, in_channels=in_channels)
            else:
                raise NotImplementedError("Not implemented.")
        else:
            return None

        # 加载预训练I3D模型权重（使用Kinetics数据集预训练）
        self_trained_flag = False
        try:
            if path_model is not None:
                if has_extra_layers:
                    self.load(model.get_i3d_model(), path_model)
                else:
                    if mode == "rgbd":
                        self.load(model, path_model, fill_dim=True)
                    else:
                        self.load(model, path_model)
        except:
            # 这意味着i3d的权重是自训练
            self_trained_flag = True
            print("self_trained")

        # 设置输出类别数
        # 注意TSM模型不需要这个函数
        model.replace_logits(self.num_classes)

        # 加载自训练权重（从我们的数据集上微调的输出结果为2类的模型）
        has_extra_layers_flag = False
        try:
            if self_trained_flag and path_model is not None:
                if has_extra_layers:
                    self.load(model.get_i3d_model(), path_model)
                else:
                    self.load(model, path_model)
        except:
            # 这意味着我们想要加载的模型有额外的层
            has_extra_layers_flag = True
            print("has_extra_layers")
        
        # 如果使用额外的层则删除I3D中没有用到的logits层
        if has_extra_layers:
            model.delete_i3d_logits()   

        # 加载额外层的自训练权重  
        if has_extra_layers_flag and path_model is not None:
            self.load(model, path_model)   

        # 加入TSM
        if self.use_tsm:
            model.add_tsm_to_i3d()

        # 是否使用GPU
        if self.use_cuda:
            if parallel:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                # Rank 1 means one machine, world_size means the number of GPUs on that machine
                dist.init_process_group("nccl", rank=rank, world_size=world_size)
                if path_model is None:
                    # Make sure that models on different GPUs start from the same initialized weights
                    torch.manual_seed(42)
                n = torch.cuda.device_count() // world_size
                device_ids = list(range(rank * n, (rank + 1) * n))
                torch.cuda.set_device(rank)
                model.cuda(rank)
                model = DDP(model.to(device_ids[0]), device_ids=device_ids)
            else:
                model.cuda()   
                         
        return model

    # 设置加载器
    def set_dataloader(self, rank, world_size, metadata_path, root_dir, transform, batch_size, parallel): 
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoDataset(metadata_path=metadata_path[phase], root_dir=root_dir, transform=transform[phase])
            if parallel:
                # 采样器
                sampler = DistributedSampler(dataset, shuffle=True, num_replicas=world_size, rank=rank)
                dataloader[phase] = DataLoader(dataset, batch_size=batch_size,
                            num_workers=int(self.num_workers/world_size),pin_memory=True, sampler=sampler)
            else:
                # 设置dataloader
                dataloader[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=self.num_workers, pin_memory=True)
        
        return dataloader
    
    # 将从数据加载器得到的标签转换成类列表
    def labels_to_list(self, labels):
        return np.argmax(labels.numpy().max(axis=2), axis=1).tolist()
    
    # 将从数据加载器得到的标签转换为类分数的列表
    def labels_to_score_list(self, labels):
        return labels.numpy().max(axis=2).tolist()

    # 转移到cuda
    def to_variable(self, v):
        if self.use_cuda:
            v = v.cuda() # 转移到GPU
        return v

    # 预测函数
    def make_pred(self, model, frames, upsample=True):
        m = model(frames)   
        if upsample == True:
            # 上采样预测帧长
            return F.interpolate(m, frames.size(2), mode="linear", align_corners=True)
        elif upsample == False:
            return m
        else:
            # 返回两个结果
            return (m, F.interpolate(m, frames.size(2), mode="linear", align_corners=True))

    # 清理平行GPU内存
    def clean_mp(self):
        if self.can_parallel:
            dist.destroy_process_group()

    # 训练拟合准备
    def fit(self,
        path_model=None, # 加载预训练或者自训练的模型地址
        model_id_suffix="", # 加在模型名字的后缀
        path_metadata_train="../data/split/metadata_train_split_0_by_camera.json",  # 训练集元数据地址
        path_metadata_validation="../data/split/metadata_validation_split_0_by_camera.json", # 验证集元数据地址
        path_metadata_test="../data/split/metadata_test_split_0_by_camera.json",    # 测试集元数据地址
        save_model_path="../data/saved_model/i3d/[model_id]/model/",                # 保存模型的地址，[model_id]将被替换
        save_tensorboard_path="../data/saved_model/i3d/[model_id]/run/",            # 保存数据的地址，[model_id]将被替换
        save_log_path="../data/saved_model/i3d/[model_id]/log/train.log",           # 保存训练日志的地址，[model_id]将被替换
        save_metada_path="../data/saved_model/i3d/[model_id]/metadata/"              # 保存元数据地址，[model_id]将被替换
    ):

        # 设置model_id
        model_id = "I3D-" + self.mode
        model_id += model_id_suffix
        # 根据不同的model_id更新保存数据与模型地址
        save_model_path = save_model_path.replace("[model_id]",model_id)
        save_tensorboard_path = save_tensorboard_path.replace("[model_id]",model_id)
        save_log_path = save_log_path.replace("[model_id]",model_id)
        save_metada_path = save_metada_path.replace("[model_id]",model_id)

        # 检查并创建save_metada_path文件夹
        check_and_create_dir(save_metada_path)
        # 将训练、验证、测试元数据复制到save_metada_path地址下
        shutil.copy(path_metadata_train,save_metada_path + "metadata_train.json")
        shutil.copy(path_metadata_validation, save_metada_path + "metadata_validation.json")
        shutil.copy(path_metadata_test,save_metada_path + "metadata_test.json")

        # 进程
        # GPU数量
        num_gpu = torch.cuda.device_count()
        if self.parallel and num_gpu > 1:
            self.can_parallel = True
            self.log("Let's use " + str(num_gpu) + "GPUs!")
            # GPU并行训练
            mp.spawn(self.fit_worker, nprocs=num_gpu,
                    args=(num_gpu, path_model, save_model_path, save_tensorboard_path, save_log_path, self.path_frame,
                    path_metadata_train, path_metadata_validation, path_metadata_test), join=True)
        else:
            self.fit_worker(0, 1, path_model, 
                save_model_path, save_tensorboard_path, save_log_path, self.path_frame,
                path_metadata_train, path_metadata_validation, path_metadata_test
                )

    # 训练拟合具体步骤
    def fit_worker(self, rank, world_size, path_model,
        save_model_path, save_tensorboard_path, save_log_path,path_frame,
        path_metadata_train, path_metadata_validation, path_metadata_test):
        # 设置日志
        # save_log_path += str(rank)
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use Inflated 3D ConvNet learner")
        self.log("save_model_path: " + save_model_path)
        self.log("save_tensorboard_path: " + save_tensorboard_path)
        self.log("save_log_path: " + save_log_path)
        self.log("path_metadata_train: " + path_metadata_train)
        self.log("path_metadata_validation: " + path_metadata_validation)
        self.log("path_metadata_test: " + path_metadata_test)
        self.log_parameters()

        # 设置模型
        model = self.set_model(rank, world_size, self.mode, 
                    path_model, self.can_parallel, phase="train")
        if model is None: return None        
        # 查看模型结构
        # for name, module in model.named_modules():
        #     print('modules:', name, module)

        # 设定数据预处理形式
        metadata_path = {"train": path_metadata_train, "validation":path_metadata_validation}
        # 对数据集进行数据增强
        transform_data = self.get_transform(self.mode, image_size=self.image_size)
        transform = {"train": transform_data, "validation": transform_data}
        if self.augment:
            transform["train"] = self.get_transform(self.mode, phase="train", image_size=self.image_size)
            # 此处打印会出现问题——NameError: name '_pil_interpolation_to_str' is not defined
            # print(transform["train"])
        # 加载数据
        dataloader = self.set_dataloader(rank, world_size, metadata_path,path_frame,
                    transform, self.batch_size_train, self.can_parallel)

        # 创建Tensorboard记录
        writer_train = SummaryWriter(save_tensorboard_path + "/train/")
        writer_validation = SummaryWriter(save_tensorboard_path + "/validation/")

        # 设置优化器
        optimizer = optim.SGD(model.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # 规定学习率衰减策略
        lr_sche = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        # 设置日志形式
        log_fm = "%s step: %d  lr: %r  loc_loss: %.4f  cls_loss: %.4f  loss: %.4f"
        
        # 训练和验证准备
        steps = 0
        epochs = 0
        nspu = self.num_steps_per_update
        nspc = self.num_steps_per_check
        nspu_nspc = nspu * nspc
        accum_gradient = {} # 累计梯度的计数器
        tot_loss = {} # 总loss
        tot_loc_loss = {} # 总定位loss
        tot_cls_loss = {} # 总分类loss
        pred_labels = {} # 预测标签
        true_labels = {} # 真实标签
        for phase in ["train", "validation"]:
            accum_gradient[phase] = 0
            tot_loss[phase] = 0.0
            tot_loc_loss[phase] = 0.0
            tot_cls_loss[phase] = 0.0
            pred_labels[phase] = []
            true_labels[phase] = []
        # 训练和验证
        while steps < self.max_steps:
            # 每个epoch都有一个训练和验证的阶段
            for phase in ["train", "validation"]:
                self.log("-"*40)
                self.log("phase " + phase)
                if phase == "train":
                    epochs += 1
                    self.log("epochs: %d  steps: %d/%d" % (epochs, steps, self.max_steps))
                    model.train(True) # 打开模型的训练模式
                    for param in model.parameters():
                        param.requires_grad = True # 打开模型的梯度更新
                else:
                    model.train(False) # 打开模型的验证模式
                    for param in model.parameters():
                        param.requires_grad = False # 关闭模型的梯度更新
                # 梯度初始化为0
                optimizer.zero_grad()
                # 对批处理数据进行迭代
                # Tqdm用来在在Python长循环中添加一个进度提示信息
                for data in tqdm.tqdm(dataloader[phase]):
                    if self.code_testing:
                        if phase == "train" and steps >= self.max_steps: break
                        if phase == "validation" and accum_gradient[phase] >= self.max_steps: break
                    accum_gradient[phase]+= 1 
                    # 将frames移动到GPU
                    frames = self.to_variable(data["frames"])
                    # 将labels转换成列表，移动到GPU
                    labels = data["labels"]
                    true_labels[phase] += self.labels_to_list(labels)
                    labels = self.to_variable(labels)
                    # 预测
                    pred = self.make_pred(model, frames)
                    # detach返回一个新的Tensor，从当前计算图中分离下来，但仍指向原变量的存放位置
                    # 新Tensor的require_grad为False，不用计算梯度
                    pred_labels[phase] += self.labels_to_list(pred.cpu().detach())
                    # 计算定位损失
                    loc_loss = F.binary_cross_entropy_with_logits(pred, labels)
                    tot_loc_loss[phase] += loc_loss.data
                    # 计算分类损失
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(pred, dim=2)[0], torch.max(labels, dim=2)[0]) 
                    tot_cls_loss[phase] += cls_loss.data
                    # 反向传播
                    loss = (0.5*loc_loss + 0.5*cls_loss) / nspu
                    tot_loss[phase] += loss.data
                    if phase == "train":
                        loss.backward()
                    # 在训练过程中积累梯度值
                    if (accum_gradient[phase] == nspu) and phase == "train":
                        steps += 1
                        if steps % nspc == 0:
                            # 记录学习率和loss值
                            lr = lr_sche.get_lr()[0]
                            tll = tot_loc_loss[phase]/nspu_nspc
                            tcl = tot_cls_loss[phase]/nspu_nspc
                            tl = tot_loss[phase]/nspc
                            self.log(log_fm % (phase, steps, lr, tll, tcl, tl))
                            # 记录到Tensorboard中
                            if rank == 0:
                                writer_train.add_scalar("localization_loss", tll, global_step=steps)
                                writer_train.add_scalar("classification_loss", tcl, global_step=steps)
                                writer_train.add_scalar("loss", tl, global_step=steps)
                                writer_train.add_scalar("learning_rate", lr, global_step=steps)
                            # 复位loss
                            tot_loss[phase] = tot_cls_loss[phase] = tot_loc_loss[phase] = 0.0
                        # 复位累计梯度计数器
                        accum_gradient[phase] = 0
                        # 更新学习率和优化器
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_sche.step()
                    # 结束循环
                if phase == "validation":
                    # 记录学习率和loss值
                    lr = lr_sche.get_lr()[0]
                    tll = tot_loc_loss[phase]/accum_gradient[phase]
                    tcl = tot_cls_loss[phase]/accum_gradient[phase]
                    tl = (tot_loss[phase]*nspu)/accum_gradient[phase]
                    # 验证集的同步损失
                    if self.can_parallel:
                        tll_tcl_tl = torch.Tensor([tll, tcl, tl]).cuda()
                        dist.all_reduce(tll_tcl_tl, op=dist.ReduceOp.SUM)
                        tll = tll_tcl_tl[0].item() / world_size
                        tcl = tll_tcl_tl[1].item() / world_size
                        tl = tll_tcl_tl[2].item() / world_size
                    self.log(log_fm % (phase, steps, lr, tll, tcl, tl))
                    # 记录到Tensorboard中
                    if rank == 0:
                        writer_validation.add_scalar("localization_loss", tll, global_step=steps)
                        writer_validation.add_scalar("classification_loss", tcl, global_step=steps)
                        writer_validation.add_scalar("loss", tl, global_step=steps)
                        writer_validation.add_scalar("learning_rate", lr, global_step=steps)
                        # 保存模型
                        self.save(model, save_model_path + str(steps) + ".pt")
                    # 重置loss
                    tot_loss[phase] = tot_loc_loss[phase] = tot_cls_loss[phase] = 0.0
                    # 复位累计梯度计数器
                    accum_gradient[phase] = 0
                    # 将precision, recall, and f-score保存到日志和Tensorboard中
                    for phase in ["train", "validation"]:
                        # 为验证集同步true_labels和pred_labels
                        if self.can_parallel and phase == "validation":
                            true_pred_labels = torch.Tensor([true_labels[phase], pred_labels[phase]]).cuda()
                            true_pred_labels_list = [torch.ones_like(true_pred_labels) for _ in range(world_size)]
                            dist.all_gather(true_pred_labels_list, true_pred_labels)
                            true_pred_labels = torch.cat(true_pred_labels_list, dim=1)
                            true_labels[phase] = true_pred_labels[0].cpu().numpy()
                            pred_labels[phase] = true_pred_labels[1].cpu().numpy()
                        self.log("Evaluate performance of phase: %s\n%s" % (phase, cr(true_labels[phase], pred_labels[phase])))
                        if rank == 0:
                            result = prfs(true_labels[phase], pred_labels[phase], average="weighted" )
                            writer = writer_train if phase == "train" else writer_validation
                            writer.add_scalar("precision", result[0], global_step=steps)
                            writer.add_scalar("recall", result[1], global_step=steps)
                            writer.add_scalar("weighted_fscore", result[2], global_step=steps)
                        # 复位
                        pred_labels[phase] = []
                        true_labels[phase] = []
    
        # 清理进程
        self.clean_mp()
        # 训练完毕
        self.log("Done training")

    # 测试
    def test(self,
            path_model=None # 加载预训练或者自训练模型的地址
            ):
        # 检查
        if path_model is None or not is_file_here(path_model):
            self.log("Need to provide a valid model path")
            return

        # 设置路径
        match = re.search(r'\bI3D-(rgb|flow)[^/]*/\b', path_model)
        model_id = match.group()[0:-1]
        if model_id is None:
            self.log("Cannot find a valid model id from the model path.")
            return
        path_root = path_model[:match.start()] + "/" + model_id + "/"
        p_metadata_test = path_root + "metadata/metadata_test.json" # metadata path (test)
        save_log_path = path_root + "log/test.log" # path to save log files
        save_viz_path = path_root + "viz/" # path to save visualizations

        # Spawn processes
        n_gpu = torch.cuda.device_count()
        if False:#self.parallel and n_gpu > 1:
            # TODO: multiple GPUs will cause an error when generating summary videos
            self.can_parallel = True
            self.log("Let's use " + str(n_gpu) + " GPUs!")
            mp.spawn(self.test_worker, nprocs=n_gpu,
                    args=(n_gpu, path_model, save_log_path, self.path_frame, save_viz_path, p_metadata_test), join=True)
        else:
            self.test_worker(0, 1, path_model, save_log_path, self.path_frame, save_viz_path, p_metadata_test)

    def test_worker(self, rank, world_size, path_model, save_log_path, path_frame, save_viz_path, p_metadata_test):
        # Set logger
        save_log_path += str(rank)
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use I3DLearner")
        self.log("Start testing with mode: " + self.mode)
        self.log("save_log_path: " + save_log_path)
        self.log("save_viz_path: " + save_viz_path)
        self.log("p_metadata_test: " + p_metadata_test)
        self.log_parameters()

        # Set model
        model = self.set_model(rank, world_size, self.mode, path_model, self.can_parallel, phase="test")
        self.log("Load model weights from " + path_model)
        if model is None: return None

        # Load dataset
        metadata_path = {"test": p_metadata_test}
        transform = {"test": self.get_transform(self.mode, image_size=self.image_size)}
        dataloader = self.set_dataloader(rank, world_size, metadata_path, path_frame,
                transform, self.batch_size_test, self.can_parallel)

        # Test
        model.train(False) # set the model to evaluation mode
        file_name = []
        true_labels = []
        pred_labels = []
        true_scores = []
        pred_scores = []
        counter = 0
        with torch.no_grad():
            # Iterate over batch data
            for d in dataloader["test"]:
                if counter % 5 == 0:
                    self.log("Process batch " + str(counter))
                counter += 1
                file_name += d["file_name"]
                frames = self.to_variable(d["frames"])
                labels = d["labels"]
                true_labels += self.labels_to_list(labels)
                true_scores += self.labels_to_score_list(labels)
                labels = self.to_variable(labels)
                pred = self.make_pred(model, frames)
                pred = pred.cpu().detach()
                pred_labels += self.labels_to_list(pred)
                pred_scores += self.labels_to_score_list(pred)

        # Sync true_labels and pred_labels for testing set
        true_labels_all = np.array(true_labels)
        pred_labels_all = np.array(pred_labels)
        true_scores_all = np.array(true_scores)
        pred_scores_all = np.array(pred_scores)

        if self.can_parallel:
            true_pred_labels = torch.Tensor([true_labels, pred_labels, true_scores, pred_scores]).cuda()
            true_pred_labels_list = [torch.ones_like(true_pred_labels) for _ in range(world_size)]
            dist.all_gather(true_pred_labels_list, true_pred_labels)
            true_pred_labels = torch.cat(true_pred_labels_list, dim=1)
            true_labels_all = true_pred_labels[0].cpu().numpy()
            pred_labels_all = true_pred_labels[1].cpu().numpy()
            true_scores_all = true_pred_labels[2].cpu().numpy()
            pred_scores_all = true_pred_labels[3].cpu().numpy()

        # Save precision, recall, and f-score to the log
        self.log("Evaluate performance of phase: test\n%s" % (cr(true_labels_all, pred_labels_all)))

        # Save roc curve and score
        self.log("roc_auc_score: %s" % str(roc_auc_score(true_scores_all, pred_scores_all, average=None)))

        # Generate video summary and show class activation map
        # TODO: this part will cause an error when using multiple GPUs
        try:
            # Video summary
            cm = confusion_matrix_of_samples(true_labels, pred_labels, n=64)
            write_video_summary(cm, file_name, path_frame, save_viz_path + str(rank) + "/")
            # Save confusion matrix
            cm_all = confusion_matrix_of_samples(true_labels, pred_labels)
            for u in cm_all:
                for v in cm_all[u]:
                    for i in range(len(cm_all[u][v])):
                        idx = cm_all[u][v][i]
                        cm_all[u][v][i] = file_name[idx]
            save_json(cm_all, save_viz_path + str(rank) + "/confusion_matrix_of_samples.json")
        except Exception as ex:
            self.log(ex)

        # Clean processors
        self.clean_mp()

        self.log("Done testing")