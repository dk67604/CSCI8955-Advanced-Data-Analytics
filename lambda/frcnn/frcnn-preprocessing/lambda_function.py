import numpy as np
import urllib
import cv2
import boto3
import pickle
import botocore
from keras_frcnn import config
import time
import os

import json

config = botocore.config.Config(connect_timeout=300, read_timeout=300)
lamda_client = boto3.client('lambda', region_name='us-east-1', config=config)
lamda_client.meta.events._unique_id_handlers['retry-config-lambda']['handler']._checker.__dict__['_max_attempts'] = 0
client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')



def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape
    new_width=1
    new_height=1
    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio,fx,fy


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio,fx,fy = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio,fx,fy


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, -1)

    # return the image
    return image





def lambda_handler(event, context):
    requestId = context.aws_request_id
    with open('config.pickle', 'rb') as f_in:
        C = pickle.load(f_in)
    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    url = event['url']
    print url
    basename = url.rsplit('/', 1)[-1].split(".")[0]
    img = url_to_image(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X, ratio,fx,fy = format_img(img, C)
    X = np.transpose(X, (0, 2, 3, 1))
    filename = '/tmp/' + basename + '.npy'
    np.save(filename, X)
    client.upload_file(filename, 'adaproject', basename + '.npy')
    payload = {}
    payload['img_process'] = basename + '.npy'
    payload['requestId'] =requestId
    timestamp = int(time.time() * 1000)
    table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])
    item = {
        'requestId': str(requestId),
        'status': 'INP',
        'createdAt': timestamp,
        'updatedAt': timestamp,
    }
    table.put_item(Item=item)
    try:
        lamda_client.invoke(FunctionName='kearsaws',
                                       InvocationType="Event",
                                       Payload=json.dumps(payload))
    except Exception as e:
        print(e)
        raise e

    response = {
        "statusCode": 200,
        "body": json.dumps(item)
    }

    return response

