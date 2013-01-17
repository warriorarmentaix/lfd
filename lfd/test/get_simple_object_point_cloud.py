from lfd import multi_item_verbs
import argparse
import numpy as np
import sys
from jds_image_proc.clouds import voxel_downsample

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_dir")
    parser.add_argument("demo_name")
    parser.add_argument("stage_num")
    parser.add_argument("file_name")
    args = parser.parse_args()
    return args

def save_simple_pc(demo_dir, demo_name, stage_num, file_name):
    verb_data_accessor = multi_item_verbs.VerbDataAccessor(test_info_dir="test/%s"%demo_dir)
    stage_info = verb_data_accessor.get_stage_info(demo_name, stage_num)
    stage_data = verb_data_accessor.get_demo_data(stage_info.stage_name)
    stage_pc = stage_data["object_clouds"][stage_info.item]["xyz"]
    pc_down = voxel_downsample(stage_pc, .02)
    np.savetxt("%s.pc" % (stage_info.item if file_name is None else file_name), pc_down)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "usage: demo directory (starting at test dir), full demo name (found in yaml file), stage number for which item is the target, point cloud file name ('.pc' will be appended)"
    else:
        args = get_args()
        save_simple_pc(args.demo_dir, args.demo_name, int(args.stage_num), args.file_name)
