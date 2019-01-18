# FaceNet特徴量を用いた顔認証精度検証

## 環境
* dockerコンテナ(172.19.32.2)で行なった
* cuda9.0
* python3.5.4

## モジュール
* facenet
* tensorflow
* sklearn
* annoy
* faiss

## インストール
1. git clone https\://github.com/davidsandberg/facenet.git
2. pip install -r requirements.txt
3. download tensorflow checkpoints
4. (pip uninstall tensorflow)
5. (pip install tensorflow-gpu)
6. $FACENET_ROOT/src配下にここのプログラムをおく

## 使い方
### データ準備
* LFWデータセット(http://vis-www.cs.umass.edu/lfw/)をダウンロードする
* mkdir data/lfw_data_10
* python split_data.py --data_dir data/lfw --out_dir data/lfw_data_10 --threshold 10
* python create_validation_json.py --data_dir data/lfw_data_10 --out_json data/lfw_data_10/validation.json

### 正解率算出
* python evaluate_face_recognition_with_json.py checkpoint --data_dir data/lfw_data_10 --validation_json data/lfw_data_10/validation.json --batch_size 1000 --method svm 

## 正解率比較
| 手法 | 正解率 [%] |
|------|------------|
| SVM | 98.61 | 
| Cosine Similarity | 97.72 | 
| Annoy (euclidean) | 97.48 | 
| Annoy (angular) | 97.55 |
| Annoy (manhattan) | 97.33 | 
| Annoy (hamming) | 1.632 | 
| Annoy (dot) | 97.44 | 
| KNN (K=5) | 98.4 | 
| KNN (K=8) | 98.43 | 
| Faiss (IndexIVFPQ) | 96.62 | 
| Faiss (IndexFlatL2) | 97.72 |

