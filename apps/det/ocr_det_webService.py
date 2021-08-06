from flask import abort
from flask import request
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from paddle_serving_app.reader import DBPostProcess, FilterBoxes
from paddle_serving_server.web_service import WebService
import config
from multiprocessing import Process
from utils import feed2Image
from initApp import app
from utils import det2rec_request

class DetService(WebService):
    def init_det(self):
        """
        this step exeute must aflter the func of load_model_config
        :return:
        """
        self.feed_var = list(self.feed_vars.keys())[0]
        self.fetch_var = list(self.fetch_vars.keys())[0]
        self.service_name = "/" + config.det_url
        self.det_preprocess = Sequential([
            ResizeByFactor(32, 960), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Transpose((2, 0, 1))
        ])
        self.filter_func = FilterBoxes(10, 10)
        self.post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "min_size": 3
        })

    def preprocess(self, feed=[], fetch=[]):
        im = feed.get('image')
        is_batch = False
        try:
            img = feed2Image(im)
        except Exception as e:
            abort(400)
        self.ori_h, self.ori_w, _ = img.shape
        det_img = self.det_preprocess(img)
        _, self.new_h, self.new_w = det_img.shape

        return {self.feed_var: det_img}, [self.fetch_var],is_batch

    def postprocess(self,feed={},fetch=[],fetch_map=None):
        det_out = fetch_map[self.fetch_var]
        ratio_list = [
            float(self.new_h) / self.ori_h, float(self.new_w) / self.ori_w
        ]
        dt_boxes_list = self.post_func(det_out, [ratio_list])
        dt_boxes = self.filter_func(dt_boxes_list[0], [self.ori_h, self.ori_w])
        return {config.det_return: dt_boxes.tolist()}

    def _launch_local_predictor(self):
        from apps.det.det_local_predict import Det_Local_Predictor
        self.client = Det_Local_Predictor()

        self.client.feed_types_[self.feed_var] = 1
        self.client.load_model_config(
            "{}".format(self.model_config),**config.det_local_predictor_config)

    def run_rpc_service(self):
        print("This API will be deprecated later. Please do not use it")
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
        def det_run():
            return self.get_prediction(request)

        self.app_instance = app_instance

    def run_debugger_service(self):

        print("web service address:")
        service_name = "/" + config.det_url
        print("http://{}:{}{}".format(config.det_ip, self.port,service_name))
        app_instance = app

        @app_instance.before_first_request
        def init():
            self._launch_local_predictor()

        @app_instance.route(service_name, methods=["POST"])
        def run():
            return self.get_prediction(request)

        self.app_instance = app_instance


    def get_prediction(self, request):
        print('json', request.json)
        if not request.json:
            abort(400,'解码错误')
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
            fetch_result = self.postprocess(
                feed=ori_feed, fetch=fetch, fetch_map=fetch_map)
            if ori_feed.get('rec') is not None:
                result = det2rec_request(ori_feed,fetch_result)
            else:
                result = {"result": fetch_result}
        except ValueError as err:
            result = {"result": str(err)}
        return result
if __name__ == '__main__':
    detservice = DetService(name=config.det_name)
    detservice.load_model_config(config.det_model_config_path)
    detservice.init_det()
    detservice.prepare_server(**config.det_prepare_server)
    detservice.run_debugger_service()
    detservice.run_web_service()
