# Create first network with Keras
import os
import boto3
import tempfile
import keras
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import numpy as np
import pickle
import keras_frcnn.resnet as nn


def handler():
    # bucket = 'my-model-bucket'
    # s3 = boto3.resource('s3')
    # s3_client = boto3.client('s3')
    #
    # # params
    #
    # f_params_file = tempfile.NamedTemporaryFile()
    # s3_client.download_file('keras-ayush', 'model_frcnn.hdf5', f_params_file.name + '.hdf5')
    # f_params_file.flush()
    # print(f_params_file.name)
    with open('config.pickle', 'rb') as f_in:
	C = pickle.load(f_in)
    class_mapping = C.class_mapping
    num_features = 1024
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)
    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)
    model_classifier = Model([feature_map_input, roi_input], classifier)
    
   # BUCKET_NAME = 'fasterrcnn_pascal2012'  # replace with your bucket name
   # KEY = 'model_frcnn.hdf5'  # replace with your object key

   # s3 = boto3.resource('s3')

   # try:
   #     s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/model_frcnn.hdf5')
   #     print 'File Found'
   # except botocore.exceptions.ClientError as e:
   #     if e.response['Error']['Code'] == "404":
   #         print("The object does not exist.")
   #     else:
   #         raise

    model_classifier.load_weights('model_frcnn.hdf5',by_name=True)
    model_classifier.compile(optimizer='sgd', loss='mse')

    print model_classifier.summary()
    return model_classifier.summary()



if __name__ == '__main__':
    handler()
