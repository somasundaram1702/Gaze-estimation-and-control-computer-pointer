
import cv2
import numpy as np
from read_and_load import read_and_load
#from openvino.inference_engine import IENetwork, IECore


class landmark(read_and_load):
    
    def predict(self, image):
        self.img = image
        net_input = self.preprocess_input(self.img)
        infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        if infer_request_handle.wait() == 0:
            self.net_output = infer_request_handle.outputs[self.output_name]
            self.x,self.y = self.find_points()
            right_eye,left_eye = self.crop_eyes(self.x,self.y)
            
            img_circle = self.draw_circle()
            return(img_circle,right_eye,left_eye,self.x,self.y)
           
    
    def find_points(self):
        h,w,_ = np.shape(self.img)
        x=[]
        y=[]
        for i,j in enumerate(np.squeeze(self.net_output)):
            if (i+1)%2 == 0:
                y.append(int(j*h))
            else:
                x.append(int(j*w))
        return(x,y)
    
    def draw_circle(self):
        for i in range(2):
            cv2.rectangle(self.img,(self.x[i]-30,self.y[i]-30),(self.x[i]+30,self.y[i]+30),(255,0,0),1)
        return(self.img) 
    
    def crop_eyes(self,x,y):
        right_eye = self.img[y[0]-15:y[0]+15,x[0]-15:x[0]+15]
        left_eye = self.img[y[1]-15:y[1]+15,x[1]-15:x[1]+15]
        return(right_eye,left_eye)
        
        
