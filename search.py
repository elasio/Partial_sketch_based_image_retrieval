import numpy as np
import cv2
import sklearn
import pickle
import P-SBIR

kmeans_dumped = pickle.load(open('image_kmeans.pkl', 'rb'))
encoder_dumped = pickle.load(open('image_encoder.pkl', 'rb'))
idf = np.load('image_idf.npy', allow_pickle = True)
image_feature = np.load('image_feature.npy', allow_pickle = True)

#search 1 query
def search_image(query_path):

    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread(query_path)
    kp, des = sift.detectAndCompute(img, None)

    des_tranform = encoder_dumped.transform( des )
    des_clusterd = kmeans_dumped.predict(des)

    hist = np.histogram(des_clustered, bins = np.arange(50001))[0]
    features = hist*idf

    score = np.dot(features, image_feature.T)
    rank_ID = np.argsort(-score)
    return rank_ID



#evaluate system
def evaluate_sketch_dataset():
    #not done yet