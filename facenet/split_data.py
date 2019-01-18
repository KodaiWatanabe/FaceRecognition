""" LFWデータセットをthresholdで分割する
--- 引数
* data_dir : lfwディレクトリまでのパス
* out_dir : コピー先ディレクトリまでのパス
* threshold : 名前ディレクトリの画像数の閾値
"""

import time
import os
import sys
import glob
import argparse
import shutil

def cp_images_to_facelog_and_facesuspect(image_path_list, facelog, facesuspect, threshold):
    sorted_image_path_list = sorted(image_path_list)
    for ei, image_path in enumerate(sorted_image_path_list):
        image_name = os.path.basename(image_path)
        if ei < threshold - 1:
            out_path = os.path.join(facesuspect, image_name)
        else:
            out_path = os.path.join(facelog, image_name)
        
        print("copy {} to {}".format(image_path, out_path))
        shutil.copyfile(image_path, out_path)

def main(args):
    
    print("+++ main")

    facelog_dir = os.path.join(args.out_dir, "FaceLog")
    facesuspect_dir = os.path.join(args.out_dir, "FaceSuspect")
    if not os.path.exists(facelog_dir):
        os.makedirs(facelog_dir)
    if not os.path.exists(facesuspect_dir):
        os.makedirs(facesuspect_dir)

    threshold = args.threshold
    more_than_threshold = 0
    name_path_list = glob.glob(os.path.join(args.data_dir, "*"))
    for name_path in name_path_list:
        name = os.path.basename(name_path)
        image_path_list = glob.glob(os.path.join(name_path, "*.*"))
        print(name, len(image_path_list)) 
        if len(image_path_list) >= threshold:
            more_than_threshold += 1
            name_facelog_dir = os.path.join(facelog_dir, name)
            name_facesuspect_dir = os.path.join(facesuspect_dir, name)
            if not os.path.exists(name_facelog_dir):
                os.makedirs(name_facelog_dir)
            if not os.path.exists(name_facesuspect_dir):
                os.makedirs(name_facesuspect_dir)
            cp_images_to_facelog_and_facesuspect(image_path_list, name_facelog_dir, name_facesuspect_dir, threshold)

    print("the number of data in LFW : ", len(name_path_list))
    print("the number of data which has more than {} images : ".format(threshold), more_than_threshold) 

    print("--- main")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="datasets/lfw") 
    parser.add_argument('--out_dir', type=str, default="datasets/lfw_data_5")
    parser.add_argument('--threshold', type=int, default=5)

    args = parser.parse_args()

    assert os.path.exists(args.data_dir), "no such LFW data directory"
    assert os.path.exists(args.out_dir), "no such out directory"

    main(args)
    
