from apps.det import DetService
from apps.rec import RecService
import config
from initApp import app
if __name__ == '__main__':
    detservice = DetService(name=config.det_name)
    detservice.load_model_config(config.det_model_config_path)
    detservice.init_det()
    detservice.prepare_server(**config.det_prepare_server)
    detservice.run_debugger_service()
    # detservice.run_web_service()

    recservice = RecService(name=config.rec_name)
    recservice.load_model_config(config.rec_model_config_path)
    recservice.init_rec()
    recservice.prepare_server(**config.rec_prepare_server)
    recservice.run_debugger_service()
    # recservice.run_web_service()
    app.run(host=config.host, port=config.port, threaded=True)