import sys
from optical_flow import *

def main(argv):
    op = OpticalFlow(rgb_vid_in_p="pos.mp4", rgb_4d_out_p="data/rgb/pos.npy",
            flow_4d_out_p="data/flow/pos.npy", save_img_dir="data/img/pos/",
            flow_threshold=0.3, sat_threshold=0.3, frame_threshold=0.01, clip_flow_bound=20,
            flow_type=1)
    op.process()
    op.contains_smoke()
    op.show_flow()

    op = OpticalFlow(rgb_vid_in_p="neg.mp4", rgb_4d_out_p="data/rgb/neg.npy",
            flow_4d_out_p="data/flow/neg.npy", save_img_dir="data/img/neg/",
            flow_threshold=0.3, sat_threshold=0.3, frame_threshold=0.01, clip_flow_bound=20,
            flow_type=1)
    op.process()
    op.contains_smoke()
    op.show_flow()

if __name__ == "__main__":
    main(sys.argv)
