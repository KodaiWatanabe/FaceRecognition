""" 検証用jsonファイルを用いた顔検索精度検証プログラム
--- 引数
* checkpoint : 学習済みモデルへのパス
* data_dir : データディレクトリまでのパス
* validation_json : validation.jsonのパス
* margin : 画像の余白
* image_size : リサイズする画像のサイズ
* batch_size : バッチ処理用
* method : 識別手法
* metric_learning : 距離学習（使わない）
"""
import time
import os
import sys
import glob
from tqdm import tqdm
import json
import argparse

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC

from scipy import misc
import numpy as np
import tensorflow as tf

import faiss
from annoy import AnnoyIndex
import facenet
import metric_learn

from augmentation_for_face import FaceAugmentation

def create_train_data(d_dict):
    train_feature_list = []
    train_label_list = []
    for d_label in d_dict.keys():
        d_feature = d_dict[d_label]
        train_feature_list.append(d_feature)
        train_label = int(d_label.split("_")[0])
        train_label_list.append(train_label)

    np_train_features = np.asarray(train_feature_list)
    np_train_labels = np.asarray(train_label_list)
    return np_train_features, np_train_labels

def calculate_accuracy_with_knn(d_dict, q_dict, metric_learning=None):

    np_train_features, np_train_labels = create_train_data(d_dict)

    if metric_learning=='lmnn':
        print("metric learning ...")
        lmnn = metric_learn.LMNN(k=5, learn_rate=1e-5, verbose=True)
        lmnn.fit(np_train_features, np_train_labels)
        np_train_features = lmnn.transform(np_train_features)
    
    print("knn training ...")
    knn = KNC(n_neighbors=8, n_jobs=16)
    knn.fit(np_train_features, np_train_labels)

    num_corrects = 0
    num_samples = 0
    for q_label in q_dict.keys():
        q_feature = q_dict[q_label].reshape(1, -1)
        if metric_learning == 'lmnn':
            lmnn.transform(q_feature)
        predictions = knn.predict(q_feature)
        label = int(q_label.split("_")[0])
        print("label : {}, prediction : {}".format(label, predictions[0]))
        if predictions[0] == label:
            num_corrects += 1
        num_samples += 1

    print("num corrects : ", num_corrects)
    print("num samples : ", num_samples)
    accuracy = 100 * float(num_corrects) / float(num_samples)
    return accuracy

def calculate_accuracy_with_svm(d_dict, q_dict):

    param_grid = [
        {'C': [1, 10, 100, 1000],
        'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
        'gamma': [0.001, 0.0001],
        'kernel': ['rbf']}
    ]

    np_train_features, np_train_labels = create_train_data(d_dict)

    #x_train, x_test, y_train, y_test = train_test_split(np_train_features, np_train_labels, test_size=0.2, random_state=0)

    print("SVM training ...")
    print("the number of train samples: ", len(np_train_features))

    svm = GridSearchCV(SVC(C=1), param_grid, cv=5, verbose=1, n_jobs=16)
    svm.fit(np_train_features, np_train_labels)
  
    print("best params: ", svm.cv_results_)

    num_corrects = 0
    num_samples = 0
    for q_label in q_dict.keys():
        q_feature = q_dict[q_label].reshape(1, -1)
        predictions = svm.predict(q_feature)
        label = int(q_label.split("_")[0])
        print("label : {} ({}), prediction : {}".format(label, q_label, predictions[0]))
        if predictions[0] == label:
            num_corrects += 1
        num_samples += 1

    print("num corrects : ", num_corrects)
    print("num samples : ", num_samples)
    accuracy = 100 * float(num_corrects) / float(num_samples)
    return accuracy

def calculate_accuracy_with_cosine_similarity(d_dict, q_dict):
    
    num_corrects = 0
    num_samples = 0
    num_log = 0
    for q_label in q_dict.keys():
        q_feature = q_dict[q_label]
        d_label_and_similarity_dict = {}
        for d_label in d_dict.keys():
            d_feature = d_dict[d_label]
            similarity = np.dot(q_feature, d_feature) / (np.linalg.norm(q_feature) * np.linalg.norm(d_feature))
            d_label_and_similarity_dict[d_label] = similarity
        
        sorted_d_label_and_similarity_list = sorted(d_label_and_similarity_dict.items(), key=lambda x: -x[1])
        prediction = sorted_d_label_and_similarity_list[0][0].split("_")[0]
        label = q_label.split("_")[0]
        print("label : ", label)
        print("prediction : ")
        for i in range(5):
            print(sorted_d_label_and_similarity_list[i][0], sorted_d_label_and_similarity_list[i][1])
        print("")
        if prediction == label:
            num_corrects += 1
        num_samples += 1

    print("num corrects : ", num_corrects)
    print("num samples : ", num_samples)
    accuracy = 100 * float(num_corrects) / float(num_samples)
    return accuracy

def calculate_accuracy_with_euclidean(d_dict, q_dict):
    
    num_corrects = 0
    num_samples = 0
    num_log = 0
    for q_label in q_dict.keys():
        q_feature = q_dict[q_label]
        d_label_and_distance_dict = {}
        for d_label in d_dict.keys():
            d_feature = d_dict[d_label]
            distance = np.sqrt(np.sum(np.square(np.subtract(q_feature, d_feature))))
            d_label_and_distance_dict[d_label] = distance
        
        sorted_d_label_and_distance_list = sorted(d_label_and_distance_dict.items(), key=lambda x: x[1])
        prediction = sorted_d_label_and_distance_list[0][0].split("_")[0]
        label = q_label.split("_")[0]
        print("label : ", label)
        print("prediction : ")
        for i in range(5):
            print(sorted_d_label_and_distance_list[i][0], sorted_d_label_and_distance_list[i][1])
        print("")
        if prediction == label:
            num_corrects += 1
        num_samples += 1

    print("num corrects : ", num_corrects)
    print("num samples : ", num_samples)
    accuracy = 100 * float(num_corrects) / float(num_samples)
    return accuracy

def calculate_accuracy_with_annoy(d_dict, q_dict):
   
    metric_name_list = ['euclidean', 'angular', "manhattan", "hamming", "dot"]
    best_accuracy = -1
    best_metric = "None"
    for metric_name in metric_name_list:
        annoy_index = AnnoyIndex(512, metric=metric_name)

        ai_to_d_label_list = []
        for ai, d_label in enumerate(d_dict.keys()):
            d_feature = d_dict[d_label]
            annoy_index.add_item(ai, d_feature)
            ai_to_d_label_list.append(d_label)

        annoy_index.build(n_trees=30)

        num_corrects = 0
        num_samples = 0
        num_log = 0
        for q_label in q_dict.keys():
            q_feature = q_dict[q_label]
            prediction_list, distance_list = annoy_index.get_nns_by_vector(q_feature, n=5,include_distances=True)
            prediction_index = prediction_list[0]
            prediction_distance = distance_list[0]
            prediction = ai_to_d_label_list[prediction_index].split("_")[0]
            label = q_label.split("_")[0]
            if prediction == label:
                num_corrects += 1
            num_samples += 1
            
        print("### {} ###".format(metric_name))
        print("num corrects : ", num_corrects)
        print("num samples : ", num_samples)
        
        accuracy = 100 * float(num_corrects) / float(num_samples)
        print("accuracy : {}%".format(accuracy))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_metric = metric_name
    return best_accuracy, best_metric

def calculate_accuracy_with_faiss(d_dict, q_dict, num_classes):
   
    fi_to_d_label_list = []
    d_feature_list = []
    base_list = []
    for fi, d_label in enumerate(d_dict.keys()):
        d_feature = d_dict[d_label]
        d_feature_list.append(d_feature)
        fi_to_d_label_list.append(d_label)
        base_list.append([fi])

    np_d_feature = np.asarray(d_feature_list)
    np_base = np.asarray(base_list)
    #indexFlatL2
    faiss_index = faiss.IndexFlatL2(512)        
    
    """ #indexIVFPQ
    quantizer = faiss.IndexFlatL2(512)
    faiss_index = faiss.IndexIVFPQ(quantizer, 512, num_classes, 64, 8)
    faiss_index.train(np_d_feature)
    """
    faiss_index.add(np_d_feature)

    num_corrects = 0
    num_samples = 0
    num_log = 0
    for q_label in q_dict.keys():
        q_feature = q_dict[q_label]
        q_feature = np.asarray([q_feature])
        distance_list, index_list = faiss_index.search(q_feature, 5)
        prediction_index = index_list[0][0]
        prediction_distance = distance_list[0]
        prediction = fi_to_d_label_list[int(prediction_index)].split("_")[0]
        label = q_label.split("_")[0]
        if prediction == label:
            num_corrects += 1
        num_samples += 1
        
    print("num corrects : ", num_corrects)
    print("num samples : ", num_samples)
    
    accuracy = 100 * float(num_corrects) / float(num_samples)
    print("accuracy : {}%".format(accuracy))

    return accuracy

def bb_to_np_image(bb, image_path, margin, image_size, face_aug=None, gamma=None):
    img = misc.imread(image_path, mode='RGB') 
    img_size = np.asarray(img.shape)[0:2]
    det = np.squeeze(np.asarray(bb))
    np_bb = np.zeros(4, dtype=np.int32)
    np_bb[0] = np.maximum(det[0]-margin/2, 0)
    np_bb[1] = np.maximum(det[1]-margin/2, 0)
    np_bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    np_bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[np_bb[1]:np_bb[3],np_bb[0]:np_bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    if face_aug!=None and gamma!=None:
        aligned = face_aug.apply_gamma_to_np_image(aligned, gamma)
    prewhitened = facenet.prewhiten(aligned)
    return prewhitened

def main(args):
    
    print("+++ main")

    validation_dict = json.load(open(args.validation_json))
    database = validation_dict['database']
    query = validation_dict['query']

    face_aug = FaceAugmentation()

    with tf.Graph().as_default():

        with tf.Session() as sess:
            facenet.load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            np_image_list = []
            tmp_d_label_list = []
            d_label_and_feature_dict = {}
            num_classes = len(database.keys())
            for d_label in tqdm(database.keys()):
                for ei, d_image_path in enumerate(database[d_label].keys()):
                    d_bb = database[d_label][d_image_path]
                    for gi, gamma in enumerate(face_aug.gamma_list):
                        np_image = bb_to_np_image(d_bb, d_image_path, args.margin, args.image_size, face_aug, gamma)
                        np_image_list.append(np_image)
                        tmp_d_label = d_label + "_{0:02d}_{1:02d}".format(ei, gi)
                        tmp_d_label_list.append(tmp_d_label)
                        if len(np_image_list) >= args.batch_size:
                            np_images = np.stack(np_image_list)
                            feed_dict = {images_placeholder: np_images, phase_train_placeholder: False}
                            embeddings_output = sess.run(embeddings, feed_dict=feed_dict)
                            for l, emb in zip(tmp_d_label_list, embeddings_output):
                                d_label_and_feature_dict[l] = emb
                            np_image_list = []
                            tmp_d_label_list = []

            if len(np_image_list) > 0:
                np_images = np.stack(np_image_list)
                feed_dict = {images_placeholder: np_images, phase_train_placeholder: False}
                embeddings_output = sess.run(embeddings, feed_dict=feed_dict)
                for l, emb in zip(tmp_d_label_list, embeddings_output):
                    d_label_and_feature_dict[l] = emb
                np_image_list = []
                tmp_d_label_list = []

            np_image_list = []
            tmp_q_label_list = []
            q_label_and_feature_dict = {}
            for q_label in tqdm(query.keys()):
                for ei, d_image_path in enumerate(query[q_label].keys()):
                    d_bb = query[q_label][d_image_path]
                    np_image = bb_to_np_image(d_bb, d_image_path, args.margin, args.image_size)
                    np_image_list.append(np_image)
                    tmp_q_label = q_label + "_{0:02d}".format(ei)
                    tmp_q_label_list.append(tmp_q_label)
                    if len(np_image_list) >= args.batch_size:
                        np_images = np.stack(np_image_list)
                        feed_dict = {images_placeholder: np_images, phase_train_placeholder: False}
                        embeddings_output = sess.run(embeddings, feed_dict=feed_dict)
                        for l, emb in zip(tmp_q_label_list, embeddings_output):
                            q_label_and_feature_dict[l] = emb
                        np_image_list = []
                        tmp_q_label_list = []

            if len(np_image_list) > 0:
                np_images = np.stack(np_image_list)
                feed_dict = {images_placeholder: np_images, phase_train_placeholder: False}
                embeddings_output = sess.run(embeddings, feed_dict=feed_dict)
                for l, emb in zip(tmp_q_label_list, embeddings_output):
                    q_label_and_feature_dict[l] = emb
                np_image_list = []
                tmp_q_label_list = []

            best_metric = None
            if args.method == 'euclidean':
                accuracy = calculate_accuracy_with_euclidean(d_label_and_feature_dict, q_label_and_feature_dict)
            if args.method == 'svm':
                accuracy = calculate_accuracy_with_svm(d_label_and_feature_dict, q_label_and_feature_dict)
            if args.method == 'knn':
                accuracy = calculate_accuracy_with_knn(d_label_and_feature_dict, q_label_and_feature_dict, metric_learning=args.metric_learning)
            if args.method == 'cosine':
                accuracy = calculate_accuracy_with_cosine_similarity(d_label_and_feature_dict, q_label_and_feature_dict)
            if args.method == 'annoy':
                accuracy, best_metric = calculate_accuracy_with_annoy(d_label_and_feature_dict, q_label_and_feature_dict)
            if args.method == 'faiss':
                accuracy = calculate_accuracy_with_faiss(d_label_and_feature_dict, q_label_and_feature_dict, num_classes)

            print("accuracy = {}%".format(accuracy))
            if best_metric != None:
                print("best metric : ", best_metric)

    print("--- main")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('model')
    parser.add_argument('--data_dir', type=str, default='data/lfw_data_6')
    parser.add_argument('--validation_json', type=str, default='data/lfw_data_6/validation.json')
    parser.add_argument('--margin', type=int, default=44)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--method', type=str, default='euclidean')
    parser.add_argument('--metric_learning', type=str, default='None')

    args = parser.parse_args()
    
    assert os.path.exists(args.data_dir), "no such data directory"
    assert os.path.exists(args.validation_json), "no such validation json file"
    assert args.method in ['euclidean', 'svm', 'knn', 'cosine', 'annoy', 'faiss'], "no such method is implemented"

    main(args)

