import cv2
import numpy as np
import sys
import os
import pickle
import pqkmeans
# extract feature of data ( data_directory = "image" || "edge || "sketch_queries" )
def feature_extraction(data_directory):
    sift = cv2.xfeatures2d.SIFT_create()

    list_des = []
    liss_name = []
    list_category = os.listdir(data_directory)
    for category in list_category:
        list_name = os.listdir(data_directory + "/" + category )
        for name in list_name:
            print (name)
            img = cv2.imread(data_directory + '/' + category + '/' + name)
            kp, des = sift.detectAndCompute(img, None)
            if des is None:
                print(name)
                continue
            list_des += [des]
            list_name = [category + '_' + name]

    np.save(data_directory + "_des.npy", np.array(list_des))
    np.save(data_directory + "_name.npy", np.array(list_name))
    print(data_directory + '>>: done extraction')

# stack all descriptor of data
def np_array_to_vStack(data_directory):
    des = np.load('image_des.npy', allow_pickle = True)
    v_stack_des = des[0]
    for i, remaining in enumerate(des[1:]):
        if remaining is None:
            continue
        print (i)
        v_stack_des = np.vstack((v_stack_des, remaining))
    np.save(data_directory + '_v_stack_des.npy',v_stack_des)

#create encoder and kmean model
def create_model(data_directory): #(data_directory = "image" || "edge")
    X = np.load(data_directory + '_v_stack_des.npy', allow_pickle = True)  
    # n_sample = X.shape[0]
    # k = n_sample/100

    encoder = pqkmeans.encoder.PQEncoder(num_subdim=4, Ks=256)
    encoder.fit(X[:50000])  # Use a subset of X for training
    print(data_directory + '>>: done train encode')

    X_pqcode = encoder.transform(X)
    np.save(data_directory + '_des_tranform_data.npy',X_pqcode )
    print(data_directory + '>>: done transform data')
    print (X_pqcode.shape)

    kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder, k=400)
    kmeans.fit(X_pqcode)
    pickle.dump(kmeans, open(data_directory + '_kmeans.pkl', 'wb'))
    pickle.dump(encoder, open(data_directory + '_encoder.pkl', 'wb'))
    print(data_directory + '>>: done create model')

#cluster descriptor of data
def cluster(data_directory): #(data_directory = "image" || "edge " || "sketch_queries")
    list_des = np.load(data_directory + "_des.npy", allow_pickle = True) 
    print(list_des.shape)
    ##################
    kmeans_dumped = pickle.load(open('image_kmeans.pkl', 'rb'))
    encoder_dumped = pickle.load(open('image_encoder.pkl', 'rb'))
    print(data_directory + '>>: done load data')

    list_des_clustered = []     #list feature clustered of each image
    for i, des in enumerate(list_des):
        des_tranform = encoder_dumped.transform( des )
        des_clusterd = kmeans_dumped.predict(des_tranform)
        list_des_clustered += [des_clusterd]
    print(data_directory + '>>: done convert')
    np.save(data_directory + "_des_clustered.npy", np.array(list_des_clustered))

#caculate histogram 
def cal_his(data_directory):
    list_des_clustered = np.load(data_directory + "_des_clustered.npy",allow_pickle = True)
    list_hist = []
    for i, des_clustered in enumerate(list_des_clustered):
        hist = np.histogram(des_clustered, bins = np.arange(401))[0]
        hist = list(hist)
        list_hist += [hist]
    list_hist = np.array(list_hist)
    print (list_hist.shape)
    print (list_hist[0].shape)
    np.save(data_directory + '_hist.npy', list_hist)
    print(data_directory + '>>: done calculator histogram')

#calculate tf-idf
def tf_idf(data_directory):
    hist = np.load(data_directory + '_hist.npy', allow_pickle = True)
    nbr_occurences = np.sum( (hist > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(hist)+1) / (1.0*nbr_occurences + 1)), 'float32')
    np.save(data_directory + '_idf.npy', idf)
    
    features = hist*idf
    np.save(data_directory + 'feature.npy', features)
    print(data_directory + '>>: done idf')


