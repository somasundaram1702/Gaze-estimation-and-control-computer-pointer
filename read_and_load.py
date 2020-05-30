
from openvino.inference_engine import IENetwork, IECore

import cv2
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class read_and_load:
    
    '''
    Class for reading and loading the model
    '''
    
    def __init__(self, model_name, model_path, device, extensions=None):
        
        self.model_weights = model_path+'.bin'
        self.model_structure = model_path+'.xml'
        self.device = device
        self.model_name = model_name
       

    def load_model(self): 
        
        try:
            self.model = IENetwork(self.model_structure,self.model_weights)
        except Exception as e:
            logging.error(f'{e}: check the path of the input model')

        
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        
        core = IECore()
        self.network = core.load_network(network=self.model,device_name=self.device, num_requests=1)
        logging.info(f'{self.model_name} model loaded successfully')
       
        
    def preprocess_input(self, image):
        input_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape(1, *input_image.shape)
        #logging.info(f'{self.model_name} preprocessing of input completed')
        return {self.input_name: input_image}


    
