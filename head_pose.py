
from read_and_load import read_and_load
from openvino.inference_engine import IENetwork, IECore
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class head_pose(read_and_load):
    
    def load_model(self): 

        self.model = IENetwork(self.model_structure,self.model_weights)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=self.model.outputs.keys()
        self.output = [i for i in self.output_name]
        core = IECore()
        self.network = core.load_network(network=self.model,device_name=self.device, num_requests=1)
        logging.info('Head pose model loaded successfully')
    
    def predict(self, image):
        self.img = image
        net_input = self.preprocess_input(self.img)
        infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        if infer_request_handle.wait() == 0:
            self.net_output = [infer_request_handle.outputs[i] for i in self.output]
        return(self.net_output)
    
