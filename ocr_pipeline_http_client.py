import numpy as np
import requests
import json
import cv2
import base64
import os
"""
#1. det请求地址
	det_url = "http://127.0.0.1:9292/det"
1.1  det 请求方式 post
1.2  请求参数 格式 json
	{"image":param0,"fetch":param1,"rec":param2}
	image：图片格式 base64 ，url ， 二进制字节码三种类型   是否必须 ： 必须
	fetch: 格式 列表 或者字符 无实际意义               是否必须 ： 必须
	rec: 格式 ：字符或者bool                           是否必须 ： 必须
	无rec参数，
	有rec参数，会直接调用rec服务返回识别的数字
	
	
2. rec 请求地址
	rec_url = "http://0.0.0.0:9292/rec"
	
2.1  det 请求方式 post

2.2  请求参数 格式 json
	{"image":param0,"points":param1,"fetch":param2}
	image：图片格式 base64 ，url ， 二进制字节码三种类型   							是否必须 ： 必须
	points: 列表 检测框的四个顶点，从左上角、右上角、右下角、左下角顺序             是否必须 ： 必须
	fetch: 字符或者列表

"""


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

det_url = "http://127.0.0.1:9292/det"
rec_url = "http://0.0.0.0:9292/rec"

path1= "./data/door2.jpg"
path2 = '7.jpg'
path0= "./data/door4.jpg"
with open(path1, 'rb') as file:
    image_data1 = file.read()
image = cv2_to_base64(image_data1)
import time
# for i in range(4):
data = {"image": image,
        'fetch':[],
        "rec":True}
headers = {'Content-Type': 'application/json;charset=UTF-8'}
r = requests.post(url=det_url, data=json.dumps(data),headers=headers)
print(r.json())
start = time.time()
rec_data = {"image": image,
            'points':[[[396.0, 640.0], [780.0, 625.0], [784.0, 748.0], [400.0, 762.0]],
                      [[201.0, 298.0], [540.0, 232.0], [560.0, 351.0], [221.0, 417.0]],
                      [[588.0, 295.0], [934.0, 250.0], [940.0, 298.0], [593.0, 344.0]],
                      [[654.0, 227.0], [841.0, 201.0], [846.0, 247.0], [660.0, 272.0]]],
            "fetch":[]
            }

# r = requests.post(url=rec_url, data=json.dumps(rec_data),headers=headers)
print(r)
print(r.json())
print(r.__dir__())
end = time.time()
print(end-start)