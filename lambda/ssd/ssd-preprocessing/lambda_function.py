import numpy as np
import urllib
import cv2
import boto3
import botocore
import time
import os

import json

config = botocore.config.Config(connect_timeout=300, read_timeout=300)
lamda_client = boto3.client('lambda', region_name='us-east-1', config=config)
lamda_client.meta.events._unique_id_handlers['retry-config-lambda']['handler']._checker.__dict__['_max_attempts'] = 0
client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')


def format_img(img):
    """ formats an image for model prediction based on config """
    img  = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
    return img


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
    url = event['url']
    print url
    basename = url.rsplit('/', 1)[-1].split(".")[0]
    print basename
    img = url_to_image(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height=img.shape[0]
    width=img.shape[1]
    X = format_img(img)
    X_float64=X.astype(np.float64)
    temp_list=[]
    temp_list.append(X_float64)
    temp_np=np.array(temp_list)
    filename = '/tmp/' + basename + '.npy'
    np.save(filename, temp_np)
    client.upload_file(filename, 'adaproject', basename + '.npy')
    payload = {}
    payload['img_process'] = basename + '.npy'
    payload['requestId'] =requestId
    payload['height'] =height
    payload['width'] =width
    timestamp = int(time.time() * 1000)
    table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])
    item = {
        'requestId': str(requestId),
        'status': 'INP',
        'createdAt': timestamp,
        'updatedAt': timestamp,
    }
    print item
    table.put_item(Item=item)
    try:
        lamda_client.invoke(FunctionName='ssd-lambda',
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

