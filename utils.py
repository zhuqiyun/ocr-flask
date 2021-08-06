import cv2
import numpy as np
from paddle_serving_app.reader import File2Image,Base64ToImage,URL2Image
from PIL import Image
from io import BytesIO
import config
import requests
import json
def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC

    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

class Byte2Img(object):
    def __init__(self):
        pass

    def __call__(self,img_b):
        img = Image.frombytes(BytesIO(img_b))
        return img.convert("RGB")


def feed2Image(im):
    if im.strip().startswith('http'):
        img = URL2Image()(im)
    elif isinstance(im, bytes):
        img = Byte2Img()(im)
    else:
        img = Base64ToImage()(im)
    return img

def det2rec_request(ori_feed,fetch_result):
    ori_feed['points'] = fetch_result.get(config.det_return)
    try:
        ret = requests.post(url=config.req_rec_url, data=json.dumps(ori_feed),headers=config.headers)
        result = ret.json()
    except Exception as e:
        fetch_result['rec'] = config.rec_err
        result = fetch_result
    return result

if __name__ == '__main__':
    img = cv2.imread('./data/door4.jpg',1)
    arr = np.asarray(img)
    dt_box = np.array([[233.0, 121.0], [450.0, 157.0], [443.0, 205.0], [226.0, 169.0]],np.float32)
    cop_img = get_rotate_crop_image(img,dt_box)
    cv2.imshow('t',cop_img)
    cv2.waitKey(0)

