import os

import paddle_serving_client.io as serving_io
path = '/home/rong/Desktop/envirenment/models'
model_path = "ch_ppocr_mobile_v2.0_rec_infer"
model_filename = "inference.pdmodel"
params_filename = "inference.pdiparams"

serving_io.inference_model_to_serving(
    dirname=os.path.join(path,model_path),
    serving_server="serving_server_rec",
    serving_client="serving_client_rec",
    model_filename=model_filename,
    params_filename=params_filename)
