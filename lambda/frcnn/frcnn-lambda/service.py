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
import json
import os
import boto3
import decimal
import time
import botocore

client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return int(obj)
        return super(DecimalEncoder, self).default(obj)


def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def handler(event, context):
    img_name = event['img_process']

    client.download_file('adaproject', img_name, '/tmp/' + img_name)
    X = np.load('/tmp/' + img_name)

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
    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    BUCKET_NAME = 'adaproject'  # replace with your bucket name
    KEY = 'model_frcnn.hdf5'  # replace with your object key

    s3 = boto3.resource('s3')

    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/model_frcnn.hdf5')
        print 'File Found'
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    model_rpn.load_weights('/tmp/model_frcnn.hdf5', by_name=True)
    model_classifier.load_weights('/tmp/model_frcnn.hdf5', by_name=True)
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    # Starting RPN prediction
    [Y1, Y2, F] = model_rpn.predict(X)
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]
    bboxes = {}
    probs = {}
    bbox_threshold = 0.8
    class_mapping = {v: k for k, v in class_mapping.items()}
    # print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded
        [P_cls, P_regr] = model_classifier.predict([F, ROIs])
        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
    final_data = []
    output = {}
    for key in bboxes:
        data = {}
        bbox = np.array(bboxes[key])
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        data[key] = {}
        for i in range(new_boxes.shape[0]):
            data[key]['x'] = str(new_boxes[i][0])
            data[key]['y'] = str(new_boxes[i][1])
            data[key]['w'] = str(new_boxes[i][2])
            data[key]['z'] = str(new_boxes[i][3])
            data[key]['prob'] = str(new_probs[i])
            final_data.append(data)

    output['bboxes'] = bboxes
    output['rpn'] = final_data
    timestamp = int(time.time() * 1000)
    table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])
    result = table.update_item(
        Key={
            'requestId': event['requestId']
        },
        ExpressionAttributeNames={
            '#status': 'status',
            '#result': 'result',
        },
        ExpressionAttributeValues={
            ':status': 'DONE',
            ':result': output,
            ':updatedAt': timestamp,
        },
        UpdateExpression='SET #status = :status, '
                         '#result = :result, '
                         'updatedAt = :updatedAt',
        ReturnValues='ALL_NEW',
    )
    response = {
        "statusCode": 200,
        "body": json.dumps(result['Attributes'],
                           cls=DecimalEncoder)
    }

    return response
