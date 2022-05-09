from optical_flow import *
import requests
import json
import os

def main():
    results = {}

    num_pos_gold_total = 0
    num_pos_not_gold_total = 0
    num_neg_gold_total = 0
    num_neg_not_gold_total = 0

    num_pos_gold_kept = 0
    num_pos_not_gold_kept = 0
    num_neg_gold_kept = 0
    num_neg_not_gold_kept = 0

    flow_thresh = 0.35
    sat_thresh = 0.35

    with open("video_labels.json") as f:
        videos = json.load(f)

    for i in videos["data"]:
        if i["label_state_admin"] == -1: continue
        url_concat = i["url_root"] + i["url_part"]
        os.system("curl -o tmp.mp4 '%s'" % url_concat)
        op = OpticalFlow(rgb_vid_in_p="tmp.mp4", flow_threshold=flow_thresh,
                sat_threshold=sat_thresh, frame_threshold=0.01, record_hsv=True)
        op.process()
        has_smoke = op.contains_smoke()
        if i["label_state_admin"] == 47:
            num_pos_gold_total += 1
            if has_smoke:
                num_pos_gold_kept += 1
        elif i["label_state_admin"] in {23, 19, 15}:
            num_pos_not_gold_total += 1
            if has_smoke:
                num_pos_not_gold_kept += 1
        elif i["label_state_admin"] == 32:
            num_neg_gold_total += 1
            if has_smoke:
                num_neg_gold_kept += 1
        elif i["label_state_admin"] in {20, 16, 12}:
            num_neg_not_gold_total += 1
            if has_smoke:
                num_neg_not_gold_kept += 1

    results["Positive Gold Standard Count"] = num_pos_gold_total
    results["Negative Gold Standard Count"] = num_neg_gold_total
    results["Positive Non-Gold Standard Count"] = num_pos_not_gold_total
    results["Negative Non-Gold Standard Count"] = num_neg_not_gold_total
    results["Positive Gold Standards Kept"] = num_pos_gold_kept
    results["Negative Gold Standards Kept"] = num_neg_gold_kept
    results["Positive Non-Gold Standards Kept"] = num_pos_not_gold_kept
    results["Negative Non-Gold Standards Kept"] = num_neg_not_gold_kept
    results["Positive Gold Retention Percent"] = (num_pos_gold_kept/num_pos_gold_total)*100
    results["Negative Gold Retention Percent"] = (num_neg_gold_kept/num_neg_gold_total)*100
    results["Positive Non-Gold Retention Percent"] = (num_pos_not_gold_kept/num_pos_not_gold_total)*100
    results["Negative Gold Retention Percent"] = (num_neg_not_gold_kept/num_neg_not_gold_total)*100

    return(results)

print(main())
