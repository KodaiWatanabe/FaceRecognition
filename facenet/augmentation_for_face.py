""" 顔認証用オーグメンテーションプログラム
--- 説明
ガンマ補正を用いた画像のオーグメンテーションをし、新しいディレクトリに保存する
--- 引数
* data_dir : 顔画像が保存されているFaceSuspectまでのパス
* out_dir : オーグメンテーション後の画像を保存するディレクトリまでのパス
"""
import time
import os
import sys
import argparse
import glob

import cv2
import numpy as np


class FaceAugmentation(object):
    
    def __init__(self):
        
        #self.gamma_list = [0.8, 0.9, 1.0, 1.1, 1.2]
        #self.gamma_list = [0.5, 1.0, 1.5]
        self.gamma_list = [1.0] # no augmentation
        self.look_up_table_list = self.create_look_up_table()

    def create_look_up_table(self):
        look_up_table_list = []
        for gamma in self.gamma_list:
            look_up_table = np.ones((256, 1), dtype='uint8') * 0
            for i in range(256):
                look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
            look_up_table_list.append(look_up_table)

        return look_up_table_list

    def augmentation_with_gamma(self, image_path, out_dir):
        img_src = cv2.imread(image_path)
        image_path_split = image_path.split("/")[-1].split(".")
        image_name = image_path_split[0]
        image_ext = image_path_split[1]
        for ei, look_up_table in enumerate(self.look_up_table_list):
            img_gamma = cv2.LUT(img_src, look_up_table)
            out_image_path = os.path.join(out_dir, image_name+"_gamma{0:02}.".format(ei)+image_ext)
            cv2.imwrite(out_image_path, img_gamma)

    def apply_gamma_to_np_image(self, np_image, gamma):
        gi = self.gamma_list.index(gamma)
        look_up_table = self.look_up_table_list[gi]
        img_gamma = cv2.LUT(np_image, look_up_table)
        return img_gamma

def main(args):
    
    print("+++ main")

    face_aug = FaceAugmentation()

    data_dir_list = glob.glob(os.path.join(args.data_dir, "*")) 
    for data_dir in data_dir_list:
        image_path_list = glob.glob(os.path.join(data_dir, "*.*"))
        for image_path in image_path_list:
            #print(image_path)
            image_out_dir_name = data_dir.split("/")[-1]
            image_out_dir = os.path.join(args.out_dir, image_out_dir_name)
            #print(image_out_dir)
            if not os.path.exists(image_out_dir):
                os.makedirs(image_out_dir)
            face_aug.augmentation_with_gamma(image_path, image_out_dir)

    print("--- main")

if __name__ == '__main__':

    home = os.path.expanduser("~") 
    work = os.path.join(home, "workspace/project/face-recognition")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join(work, "1_顔のデータと構造/FaceSuspect"))
    parser.add_argument('--out_dir', type=str, default="FaceSuspectGamma")
    
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        print("argument 'out_dir' '{}' already exists.".format(args.out_dir))
        select = input("Do you wanna continue? (y/n)")
        if select!='y':
            sys.exit(1)
    else:
        print("no such directory '{}'. so now make the directory".format(args.out_dir))
        os.makedirs(args.out_dir)

    main(args)

