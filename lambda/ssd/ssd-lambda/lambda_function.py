# Create first network with Keras
import os
import boto3
import tempfile
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import pickle
import json
import decimal
import time
import botocore

import tensorflow as tf
from ssd import SSD300
from ssd_utils import BBoxUtility

client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1

def claculate_bbox_and_prob(results,height,width):
    bboxes = {}
    probs = {}
    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    for i in range(top_conf.shape[0]):
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        bboxes[label_name] = []
        probs[label_name] = []

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * width))
        ymin = int(round(top_ymin[i] * height))
        xmax = int(round(top_xmax[i] * width))
        ymax = int(round(top_ymax[i] * height))
	label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        score = top_conf[i]
        bboxes[label_name].append([xmin, ymin, xmax, ymax])
        probs[label_name].append(decimal.Decimal(score*100))
    return bboxes,probs

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return int(obj)
        return super(DecimalEncoder, self).default(obj)



def lambda_handler(event, context):
    img_name = event['img_process']
    height=int(event['height'])
    width=int(event['width'])

    client.download_file('adaproject', img_name, '/tmp/' + img_name)
    X = np.load('/tmp/' + img_name)

    with open('config.pickle', 'rb') as f_in:
        C = pickle.load(f_in)
        class_mapping = C.class_mapping
        class_mapping = {v: k for k, v in class_mapping.items()}
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}


    BUCKET_NAME = 'adaproject'  # replace with your bucket name
    KEY = 'weights_SSD300.hdf5'  # replace with your object key

    s3 = boto3.resource('s3')

    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/weights_SSD300.hdf5')
        print 'Model Found'
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    input_shape = (300, 300, 3)
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('/tmp/weights_SSD300.hdf5', by_name=True)
    bbox_util = BBoxUtility(NUM_CLASSES)
    inputs_single = preprocess_input(X)
    preds_single = model.predict(inputs_single, batch_size=1, verbose=0)
    results = bbox_util.detection_out(preds_single)
    bboxes,probs=claculate_bbox_and_prob(results=results,height=height,width=width)
    final_data = {}
    final_data['bboxes']=bboxes
    final_data['probs']=probs
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
            ':result': final_data,
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
