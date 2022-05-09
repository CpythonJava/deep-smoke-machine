import sys
from utils.util import *
from bin.i3d_learner_self import I3d

# This is the main script for model testing
# For detailed usage, run terminal command "sh bg.sh"
def main(argv):
    if len(argv) < 3:
        print("Usage: python test.py [method] [model_path]")
        return
    method = argv[1]
    model_path = argv[2]
    if method is None or model_path is None:
        print("Usage: python test.py [method] [model_path]")
        return
    for i in range(len(argv)-2):
        # Can chain at most three model paths
        # BUG: when putting too many model paths at once, creating dataloader becomes very slow
        test(method=method, model_path=argv[2+i])


def test(method=None, model_path=None):
    if method == "i3d-rgb":
        cv("rgb", "i3d", model_path, augment=True, perturb=False)
    else:
        print("Method not allowed")
        return


# Cross validation of different models
def cv(mode, method, model_path, augment=True, perturb=False):
    if mode == "rgb":
        path_frame = "../data/rgb/"
    elif mode == "flow":
        path_frame = "../data/flow/"

    # Set the model based on the desired method
    # The training script "train.py" has descriptions about these methods
    if method == "i3d":
        model = I3d(mode=mode, augment=augment, path_frame=path_frame)
    else:
        print("Method not allowed.")
        return
    model.test(path_model=model_path)


if __name__ == "__main__":
    main(sys.argv)