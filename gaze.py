
import logging
from read_and_load import read_and_load
from openvino.inference_engine import IENetwork, IECore
import cv2

class gaze(read_and_load):
    
    def load_model(self): 
        self.input_shapes=[]
        self.model = IENetwork(self.model_structure,self.model_weights)
        self.input_names=[i for i in iter(self.model.inputs)]
        for i in self.input_names:
            self.input_shapes.append(self.model.inputs[i].shape)
            
        self.output_name = next(iter(self.model.outputs))
      
        core = IECore()
        self.network = core.load_network(network=self.model,device_name=self.device, num_requests=1)
        #logging.info('Gaze model loaded successfully')
        
        
    def preprocess_input(self, left_eye_image,right_eye_image,head_pose_angles):
        #print(np.shape(left_eye_image))
        left_eye_image = cv2.resize(left_eye_image, (self.input_shapes[1][3], self.input_shapes[1][2]))
        right_eye_image = cv2.resize(right_eye_image, (self.input_shapes[1][3], self.input_shapes[1][2]))
        
        left_eye_image = left_eye_image.transpose((2, 0, 1))
        right_eye_image = right_eye_image.transpose((2, 0, 1))
        
        left_eye_image = left_eye_image.reshape(1, *left_eye_image.shape)
        right_eye_image = right_eye_image.reshape(1, *right_eye_image.shape)
        #logging.info(f'{self.model_name} preprocessing input completed')
        return {self.input_names[0]:head_pose_angles,self.input_names[1]:left_eye_image,self.input_names[2]:right_eye_image}
        
    
    def predict(self, left_eye_image,right_eye_image,head_pose_angles):
        #self.img = image
        net_input = self.preprocess_input(left_eye_image,right_eye_image,head_pose_angles)
        infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        if infer_request_handle.wait() == 0:
            self.net_output = infer_request_handle.outputs[self.output_name]
        return(self.net_output)
