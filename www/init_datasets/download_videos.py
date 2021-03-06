import sys
import urllib.request
sys.path.append("..")
from utils.util import *


# Download all videos in the metadata json file
def main(argv):
    vm = load_json("../../data/metadata.json")
    video_root_path = "../../data/videos/"
    check_and_create_dir(video_root_path)
    problem_video_ids = []
    for v in vm:
        # Do not download videos with bad data
        if v["label_state"] == -2 or v["label_state_admin"] == -2:
            continue
        file_path = video_root_path + v["file_name"] + ".mp4"
        if is_file_here(file_path): continue # skip if file exists
        print("Download video", v["id"])
        try:
            urllib.request.urlretrieve(v["url_root"] + v["url_part"], file_path)
        except:
            print("\tError downloading video", v["id"])
            problem_video_ids.append(v["id"])
    print("Done download_videos.py")
    if len(problem_video_ids) > 0:
        print("The following videos were not downloaded due to errors:")
        for i in problem_video_ids:
            print("\ti\n")


if __name__ == "__main__":
    main(sys.argv)
