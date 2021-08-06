det_model_config_path = "det_serving_server"
det_local_predictor_config = {
'use_gpu':False,
'gpu_id':0,
'use_profile':False,
'thread_num':1,
'mem_optim':True,
'ir_optim':False,
'use_trt':False,
'use_lite':False,
'use_xpu':False,
'use_feed_fetch_ops':False
}
det_ip = '0.0.0.0'
det_url = "det"
det_name = 'det'
det_prepare_server = {
"workdir" : 'workdir',
"port": 9292,
"device": "cpu"
}
det_return = 'dt_boxes'

#####rec 配置
headers = {'Content-Type': 'application/json;charset=UTF-8'}

rec_model_config_path = "rec_serving_server"
rec_local_predictor_config = {
'use_gpu':False,
'gpu_id':0,
'use_profile':False,
'thread_num':1,
'mem_optim':True,
'ir_optim':False,
'use_trt':False,
'use_lite':False,
'use_xpu':False,
'use_feed_fetch_ops':False
}
rec_ip = '0.0.0.0'
rec_url = "rec"
rec_name = 'rec'
rec_prepare_server = {
"workdir" : 'workdir',
"port": 9292,
"device": "cpu"
}
rec_character_dict_path= 'conf_files/ppocr_keys_v1.txt'
rec_character_type='ch'
rec_use_space_char=False
rec_drop_score = 0.5
req_rec_url = "http://{}:{}/{}".format(rec_ip,rec_prepare_server.get('port'),rec_name)
rec_err = "rec服务故障"
host = "0.0.0.0"
port = 9292

