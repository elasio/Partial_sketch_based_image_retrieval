import P-SBIR
import cv2
import numpy as np
import sys
import os
import pickle

#process image data
feature_extraction('image')
np_array_to_vStack('image')
create_model('image')
cluster('image')
cal_his('image')
tf_idf('image')

#process edge data
feature_extraction('edge')
np_array_to_vStack('edge')
create_model('edge')
cluster('edge')
cal_his('edge')
tf_idf('edge')

#process sketch query data
feature_extraction('sketch_queries')
cluster('sketch_queries')
cal_his('sketch_queries')
tf_idf('edge')