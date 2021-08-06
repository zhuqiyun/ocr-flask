from paddle_serving_app.reader import OCRReader
from flask import Flask
from flask import request
from flask import abort
import numpy as np
from paddle_serving_app.reader import SortedBoxes
from utils import feed2Image
from utils import get_rotate_crop_image
from paddle_serving_server.web_service import WebService
from apps.rec.rec_postprcess import CTCLabelDecode
import copy
import config
from initApp import app
from multiprocessing import Process



class RecService(WebService):
    def init_rec(self):
        self.ocr_reader = OCRReader(char_dict_path=config.rec_character_dict_path)
        self.sorted_boxes = SortedBoxes()
        self.feed_var = list(self.feed_vars)[0]
        self.fetch_var = list(self.fetch_vars)[0]
        print("path",config.rec_character_dict_path)
        self.ctc_decoder = CTCLabelDecode(
                character_dict_path=config.rec_character_dict_path,
                character_type=config.rec_character_type,
                use_space_char=config.rec_use_space_char,)
        self.service_name = "/" + config.rec_url

    def preprocess(self, feed=[], fetch=[]):
        # TODO: to handle batch rec images
        im = feed['image']
        img = feed2Image(im)
        dt_boxes = np.array(feed['points'],np.float32)
        self.sorted_boxes(dt_boxes)
        img_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_list.append(img_crop)

        max_wh_ratio = 0
        for i, boximg in enumerate(img_list):
            h, w = boximg.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        _, w, h = self.ocr_reader.resize_norm_img(img_list[0],
                                                  max_wh_ratio).shape
        imgs = np.zeros((len(img_list), 3, w, h)).astype('float32')
        norm_img_batch = []
        for i, img in enumerate(img_list):
            norm_img = self.ocr_reader.resize_norm_img(img, max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()
        feed = {self.feed_var: norm_img_batch,"img_list":img_list,'points':dt_boxes}
        fetch = [self.fetch_var]
        return feed, fetch, True

    def _launch_local_predictor(self):
        from apps.rec.rec_local_predict import Rec_Local_Predictor
        self.client = Rec_Local_Predictor()

        self.client.feed_types_[self.feed_var] = 1
        self.client.load_model_config(
            "{}".format(self.model_config),**config.det_local_predictor_config)

    def run_rpc_service(self):
        print("This API will be deprecated later. Please do not use it")
        import socket
        localIP = socket.gethostbyname(socket.gethostname())
        print("web service address:")
        service_name = "/" + config.rec_url
        print("http://{}:{}{}".format(config.rec_ip, self.port, service_name))

        p_rpc = Process(target=self._launch_rpc_service)
        p_rpc.start()

        app_instance = app

        @app_instance.before_first_request
        def init():
            self._launch_web_service()

        service_name = self.service_name

        @app_instance.route(service_name, methods=["POST"])
        def run():
            return self.get_prediction(request)

        self.app_instance = app_instance

    def run_debugger_service(self):

        print("web service address:")
        service_name = self.service_name
        print("http://{}:{}{}".format(config.rec_ip, self.port,service_name))
        app_instance = app

        @app_instance.before_first_request
        def init():
            self._launch_local_predictor()

        @app_instance.route(service_name, methods=["POST"])
        def rec_run():

            return self.get_prediction(request)
        self.app_instance = app_instance

    def get_prediction(self, request):
        print('json', request.json)
        if not request.json:
            abort(400)
        if "fetch" not in request.json:
            abort(400)
        try:
            ori_feed = request.json
            feed, fetch, is_batch = self.preprocess(ori_feed)
            if isinstance(feed, dict) and "fetch" in feed:
                del feed["fetch"]
            if len(feed) == 0:
                raise ValueError("empty input")
            fetch_map = self.client.predict(
                feed=feed, fetch=fetch, batch=is_batch)

            result = self.postprocess(
                feed=feed, fetch=fetch, fetch_map=fetch_map)
            result = {"result": result}
        except ValueError as err:
            result = {"result": str(err)}
        return result


    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        dt_boxes =feed['points']
        img_list = feed['img_list']
        img_num = len(img_list)
        beg_img_no = 0
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))

        preds = fetch_map[self.fetch_var]
        rec_res = [['', 0.0]] * img_num

        rec_result = self.ctc_decoder(preds)

        for rno in range(len(rec_result)):
            rec_res[indices[rno]] = rec_result[rno]
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            if isinstance(box,np.ndarray):
                box = box.tolist()
            text, score = rec_reuslt
            if score >= config.rec_drop_score:
                filter_boxes.append(box)
                filter_rec_res.append([box,text,str(score)])
        result = {"result": filter_rec_res}
        return result

if __name__ == '__main__':
    recservice = RecService(name=config.rec_name)
    recservice.load_model_config(config.rec_model_config_path)
    recservice.init_rec()
    recservice.prepare_server(**config.rec_prepare_server)
    recservice.run_debugger_service()
    recservice.run_web_service()