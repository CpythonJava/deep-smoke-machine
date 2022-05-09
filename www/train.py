import sys
sys.path.append("..")
from bin.i3d_learner import I3d

def main(argv):
    if len(argv) < 2:
        print("The format is : python train.py [method]")
        return 
    method = argv[1]
    model_path = None
    train(method=method, model_path=model_path)


def train(method=None, model_path=None):
    '''训练函数'''  
    if method == 'ssl-i3d-rgb':
        if model_path == None:
            model_path = "model/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("semi-sup", "rgb", "i3d", model_path=model_path, augment=True, perturb=False)
    else:
        print("Method is not allowed!")
        return

def cv(sup_mode, mode, model_name, model_path=None, augment=True, perturb=False):
    '''训练部署'''

    # 判断监督学习方式
    if sup_mode == "semi-sup":
        print("It is semi-supervised.")
    elif sup_mode == "full-sup":
        print("It is full-supervised.")
    else:
        print("It is a wrong supervised mode!")
        return

    # 判断训练模式
    if mode == "rgb":
        path_frame = "../data/rgb/"
    else:
        print("It is a wrong mode!")
        return
    
    # 判断使用模型
    if model_name == "i3d":
        # 设置初始学习率
        init_lr = 0.01
        # 设置学习率衰减拐点
        milestones = [2000, 4000, 8000, 10000]
        # 最大训练步数
        max_steps = 12000
        # 并行进程数
        num_workers = 0
        # 创建模型对象
        model = I3d(
            mode=mode,
            max_steps= max_steps,
            augment=augment,
            path_frame=path_frame,
            init_lr=init_lr,
            milestones=milestones,
            num_workers=num_workers
        )

    else:
        print("It is a wrong model!")
        return

    # 将模型地址运用到所有切分数据集
    if type(model_path) is not list:
        model_path = [model_path]*8

    # 模型对象训练
    model.fit(
        path_model=model_path[6],
        model_id_suffix="-s6",
        path_metadata_train="../data/split/metadata_train_random_split_by_camera.json",
        path_metadata_validation="../data/split/metadata_validation_random_split_by_camera.json",
        path_metadata_test="../data/split/metadata_test_random_split_by_camera.json"
    )
    
if __name__ == "__main__":
    main(sys.argv)
