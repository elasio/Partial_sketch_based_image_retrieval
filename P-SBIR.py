import cv2
import numpy as np
import sys
import os
import pickle
def feature_extraction(data_directory): #(data_directory = "image" || "edge || "sketch_queries")
    sift = cv2.xfeatures2d.SIFT_create()

    list_des = []
    liss_name = []
    list_category = os.listdir(data_directory)
    for category in list_category:
        list_name = os.listdir(data_directory + "/" + category )
        for name in category:
            print (name)
            img = cv2.imread(data_directory + '/' + category + '/' + name)
            kp, des = sift.detectAndCompute(img, None)
            if len(des) == 0:
                print(name)
                continue
            list_des += [des]
            list_name = [category + '_' + name]

    np.save(data_directory + "_des.npy", np.array(list_des))
    np.save(data_directory + "_name.npy", np.array(list_name))
    print('done' + data_directory + 'extraction')

def create_model(data_directory): #(data_directory = "image" || "edge")
    X = np.load(data_director + 'v_stack_des.npy', allow_pickle = True)  
    # n_sample = X.shape[0]
    # k = n_sample/100

    encoder = pqkmeans.encoder.PQEncoder(num_subdim=4, Ks=256)
    encoder.fit(X[:50000])  # Use a subset of X for training
    print('done train encode')

    X_pqcode = encoder.transform(X)
    np.save(data_directory + '_des_tranform_data.npy',X_pqcode )
    print('done transform data')
    print (X_pqcode.shape)

    kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder, k=50000)
    kmeans.fit(X_pqcode)
    pickle.dump(kmeans, open(data_directory + '_kmeans.pkl', 'wb'))
    pickle.dump(encoder, open(data_directory + '_encoder.pkl', 'wb'))
    print('done create model' + data_directory)

def cluster(data_directory) #(data_directory = "image" || "edge " || "sketch_queries")
    list_des = np.load(data_directory + "_des.npy", allow_pickle = True) 
    print(list_des.shape)
    ##################
    kmeans_dumped = pickle.load(open('image_kmeans.pkl', 'rb'))
    encoder_dumped = pickle.load(open('image_encoder.pkl', 'rb'))
    print('done load data')

    list_des_clustered = []     #list feature clustered of each image
    for i, des in enumerate(list_des):
        des_tranform = encoder_dumped.transform( des )
        des_clusterd = kmeans_dumped.predict(sketch_queries_des_tranform)
        list_des_clustered += [des_clusterd]
    print('done convert' + data_directory)
    np.save(data_directory + "des_clusterd.npy", np.array(list_des_clustered))

def cal_his(data_directory):
    list_des_clustered = np.load(data_directory + "des_clusterd.npy",allow_pickle = True)
    list_hist = []
    for i, des_clustered in enumerate(list_des_clustered):
        hist = np.histogram(des_clustered, bins = np.arange(50001))[0]
        list_hist += [hist]
    list_hist = np.array(list_hist)
    print (list_hist.shape)
    print (list_hist[0].shape)
    np.save(data_directory + '_hist.npy', list_hist)
    print('done calculator histogram' + data_directory)

#process image data
feature_extraction('image')
create_model('image')
cluster('image')
cal_his('image')

#process edge data
feature_extraction('edge')
create_model('edge')
cluster('edge')
cal_his('edge')

#process sketch query data
feature_extraction('sketch_queries')
cluster('sketch_queries')
cal_his('sketch_queries')
