""" 検証用jsonファイルを作成するプログラム
--- 引数
* data_dir : dataディレクトリまでのパス
* out_json : 検証用jsonファイルのパス
"""
import time
import os
import sys
import argparse
import glob
import json
import copy
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from scipy import misc
import align.detect_face

def detect_face_from_image_path(image_path, pnet, rnet, onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    img = misc.imread(image_path, mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        return "NoFace"
    bouding_box = bounding_boxes[0,0:4]
    confidence = bounding_boxes[0,4]
    if confidence < 0.85:
        print("################# no confidende : ", confidence)
    if len(bounding_boxes) > 1:
        #print("################ multi faces", image_path, len(bounding_boxes), bounding_boxes[:,4])
        if bounding_boxes[0,4]-bounding_boxes[1,4] < 0.0001:
            print("not good sample")
            return "NoFace"

    return bouding_box

def main(args):
    
    print("+++ main")
    
    gpu_memory_fraction = 1.0 

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        facelog_dir_list = glob.glob(os.path.join(args.data_dir, "FaceLog", "*"))
        facesuspect_dir_list = glob.glob(os.path.join(args.data_dir, "FaceSuspect", "*"))

        validation_dict = {}
        validation_dict['database'] = {}
        validation_dict['query'] = {}

        print("process FaceLog ...")
        facelog_id = 0
        for facelog_dir in tqdm(facelog_dir_list):
            image_path_list = glob.glob(os.path.join(facelog_dir, "*.*"))
            
            for image_path in image_path_list:    
                bb = detect_face_from_image_path(image_path, pnet, rnet, onet)
                if bb == 'NoFace':
                    continue
                facelog_label = "{0:04d}_".format(facelog_id) + os.path.basename(facelog_dir)
                if not facelog_label in validation_dict['query'].keys():
                    validation_dict['query'][facelog_label] = {}
                validation_dict['query'][facelog_label][image_path] = list(bb)
            facelog_id += 1

        print("process FaceSuspect ...")
        facesuspect_id = 0
        for facesuspect_dir in tqdm(facesuspect_dir_list):
            image_path_list = glob.glob(os.path.join(facesuspect_dir, "*.*"))
            
            for image_path in image_path_list:    
                bb = detect_face_from_image_path(image_path, pnet, rnet, onet)
                if bb == 'NoFace':
                    continue
                facesuspect_label = "{0:04d}_".format(facesuspect_id) + os.path.basename(facesuspect_dir)
                if not facesuspect_label in validation_dict['database'].keys():
                    validation_dict['database'][facesuspect_label] = {}
                validation_dict['database'][facesuspect_label][image_path] = list(bb)
            facesuspect_id += 1

    out_json_file = open(args.out_json, "w")
    json.dump(validation_dict, out_json_file)

    print("--- main")

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/lfw_data_6')
    parser.add_argument('--out_json', type=str, default='data/lfw_data_6/validation.json')

    args = parser.parse_args()

    assert os.path.exists(args.data_dir), "no such data directory"

    main(args)

    print("END")
