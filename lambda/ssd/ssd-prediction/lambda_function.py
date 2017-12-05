import numpy as np
import urllib
import cv2
import boto3
import pickle
import botocore
import time
import os
from keras_frcnn import config
from sklearn.metrics import average_precision_score
import json
import decimal
import xml.etree.ElementTree as ET

config = botocore.config.Config(connect_timeout=300, read_timeout=300)
lamda_client = boto3.client('lambda', region_name='us-east-1', config=config)
lamda_client.meta.events._unique_id_handlers['retry-config-lambda']['handler']._checker.__dict__['_max_attempts'] = 0
client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')


def get_annotaion(annot_name):
    client.download_file('adaproject', annot_name, '/tmp/' + annot_name + '.xml')
    et = ET.parse('/tmp/' + annot_name + '.xml')
    element = et.getroot()
    element_objs = element.findall('object')
    annotation_data = {'bboxes': []}

    if len(element_objs) > 0:
        for element_obj in element_objs:
            class_name = element_obj.find('name').text
            obj_bbox = element_obj.find('bndbox')
            x1 = int(round(float(obj_bbox.find('xmin').text)))
            y1 = int(round(float(obj_bbox.find('ymin').text)))
            x2 = int(round(float(obj_bbox.find('xmax').text)))
            y2 = int(round(float(obj_bbox.find('ymax').text)))
            difficulty = int(element_obj.find('difficult').text) == 1
            annotation_data['bboxes'].append(
                {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
    return annotation_data


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def calc_iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_map(pred, gt):
    T = {}
    P = {}

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1']
            gt_x2 = gt_box['x2']
            gt_y1 = gt_box['y1']
            gt_y2 = gt_box['y2']
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = calc_iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    return T, P


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return int(obj)
        return super(DecimalEncoder, self).default(obj)


def format_img(img):
    """ formats an image for model prediction based on config """
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
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
    table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])
    result = table.get_item(
        Key={
            'requestId': event['requestId']
        }
    )
    dbresult = result['Item']
    status = dbresult['status']
    detection_result = {}
    if status == 'DONE':
        with open('config.pickle', 'rb') as f_in:
            C = pickle.load(f_in)
        class_mapping = C.class_mapping
        class_mapping = {v: k for k, v in class_mapping.items()}
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        detection = dbresult['result']
        bboxes = detection['bboxes']
        probs = detection['probs']
        url = event['url']
        basename = url.rsplit('/', 1)[-1].split(".")[0]
        img = url_to_image(url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        real_dets = []
        T = {}
        P = {}
        for key in bboxes:
            bbox = np.array(bboxes[key])
            lower_key = key.lower()
            new_probs = np.array(probs[key])
            print new_probs
            for jk in range(bbox.shape[0]):
                (real_x1, real_y1, real_x2, real_y2) = bbox[jk, :]
                real_det = {'x1': real_x1, 'x2': real_x2, 'y1': real_y1, 'y2': real_y2, 'class': key,
                            'prob': new_probs[jk]}
                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (
                int(class_to_color[lower_key][0]), int(class_to_color[lower_key][1]),
                int(class_to_color[lower_key][2])), 2)

                textLabel = '{}: {}'.format(lower_key, int(new_probs[jk]))

                real_dets.append(real_det)

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                textOrg = (real_x1, real_y1 + 10)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        maP = None
        if 'annotation' in event:
            annot_name = event['annotation']
            annotation_data = get_annotaion(annot_name)
            t, p = get_map(real_dets, annotation_data['bboxes'])
            for key in t.keys():
                if key not in T:
                    T[key] = []
                    P[key] = []
                T[key].extend(t[key])
                P[key].extend(p[key])
            all_aps = []
            for key in T.keys():
                ap = average_precision_score(T[key], P[key])
                print('{} AP: {}'.format(key, ap))
                all_aps.append(ap)
            maP = np.mean(np.array(all_aps))
        if maP is not None:
            detection_result['maP'] = maP
        detection_result['real_dets'] = real_dets
        image_to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/tmp/' + basename + '_final' + '.jpg', image_to_write)
        client.upload_file('/tmp/' + basename + '_final' + '.jpg', 'adaproject', basename + '_final.jpg')
        detection_result['image'] = basename + '_final.jpg'
        detection_result['status'] = status
        detection_result['requestId'] = event['requestId']
        # create a response
        response = {
            "statusCode": 200,
            "body": json.dumps(detection_result,
                               cls=DecimalEncoder)
        }

        return response
    else:
        detection_result['status'] = status
        detection_result['requestId'] = event['requestId']
        response = {
            "statusCode": 200,
            "body": json.dumps(detection_result,
                               cls=DecimalEncoder)
        }
        return response
