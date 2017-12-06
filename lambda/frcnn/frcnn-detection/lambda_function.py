import numpy as np
import urllib
import cv2
import boto3
import pickle
import botocore
from keras_frcnn import config
import decimal
import xml.etree.ElementTree as ET
from sklearn.metrics import average_precision_score
import json
import os
import math

config = botocore.config.Config(connect_timeout=300, read_timeout=300)
lamda_client = boto3.client('lambda', region_name='us-east-1', config=config)
lamda_client.meta.events._unique_id_handlers['retry-config-lambda']['handler']._checker.__dict__['_max_attempts'] = 0
client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return int(obj)
        return super(DecimalEncoder, self).default(obj)


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape
    new_width = 1
    new_height = 1
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
    return img, ratio, fx, fy


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


def get_map(pred, gt, f):
    T = {}
    P = {}
    fx, fy = f

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
            gt_x1 = gt_box['x1'] / fx
            gt_x2 = gt_box['x2'] / fx
            gt_y1 = gt_box['y1'] / fy
            gt_y2 = gt_box['y2'] / fy
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

    # import pdb
    # pdb.set_trace()
    return T, P


def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


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
        url = event['url']
        basename = url.rsplit('/', 1)[-1].split(".")[0]
        img = url_to_image(url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        X, ratio, fx, fy = format_img_size(img, C)
        all_dets = []
        temp = dbresult['result']
        index = 0
        rpn_class = {}
        rpn_prob = {}
        for item in temp['rpn']:
            for key in item:
                temp_box = np.zeros((1, 4), dtype=np.int64)
                temp_prob = np.zeros(shape=(1), dtype=np.float64)
                for i in item[key]:

                    temp1 = item[key]
                    if i == 'x':
                        temp_box[index][0] = temp1[i]
                    if i == 'y':
                        temp_box[index][1] = temp1[i]
                    if i == 'w':
                        temp_box[index][2] = temp1[i]
                    if i == 'z':
                        temp_box[index][3] = temp1[i]
                    if i == 'prob':
                        temp_prob[index] = temp1[i]

                rpn_class[key] = temp_box
                rpn_prob[key] = temp_prob
        print rpn_class
        key_boxes = temp['bboxes']
        T = {}
        P = {}
        real_dets = []
        for key in rpn_class:
            temp_box = rpn_class[key]
            temp_prob = rpn_prob[key]
            for jk in range(temp_box.shape[0]):
                (x1, y1, x2, y2) = temp_box[jk, :]
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': temp_prob[jk]}
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                real_det = {'x1': real_x1, 'x2': real_x2, 'y1': real_y1, 'y2': real_y2, 'class': key,
                            'prob': temp_prob[jk]}
                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                              2)

                textLabel = '{}: {}'.format(key, float(100 * temp_prob[jk]))
                all_dets.append(det)
                real_dets.append(real_det)

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)
                print textOrg
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        maP = None
        if 'annotation' in event:
            annot_name = event['annotation']
            annotation_data = get_annotaion(annot_name)
            t, p = get_map(all_dets, annotation_data['bboxes'], (fx, fy))
            for key in t.keys():
                if key not in T:
                    T[key] = []
                    P[key] = []
                T[key].extend(t[key])
                P[key].extend(p[key])
            all_aps = []
            count = 0
            for key in T.keys():
                ap = average_precision_score(T[key], P[key])
                print('{} AP: {}'.format(key, ap))

                count += 1
                all_aps.append(ap)
            maP = np.nansum(np.array(all_aps))
            maP=maP/count
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
