
import cv2
import numpy as np
import time
#from openvino.inference_engine import IENetwork, IECore
import sys
import os
import argparse
from read_and_load import read_and_load

class face_detect(read_and_load):
    '''
    Class for the Face Detection Model.
    '''

    def predict(self, image_inp):
      
        net_input = self.preprocess_input(image_inp)
        infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        if infer_request_handle.wait() == 0:
            net_output = infer_request_handle.outputs[self.output_name]
            #print(np.shape(net_output))
            boxes = self.preprocess_output(net_output)
        return self.draw_outputs(boxes, image_inp)


        
    def draw_outputs(self, coords, image):
 
        w = image.shape[1]
        h = image.shape[0]
        boxes = []
        for box in coords:
            p1 = (int(box[0] * w), int(box[1] * h))
            p2 = (int(box[2] * w), int(box[3] * h))
            boxes.append([p1[0], p1[1], p2[0], p2[1]])
            image = cv2.rectangle(image, p1, p2, (0, 0, 255), 3)
            crp_img = image[p1[1]:p2[1],p1[0]:p2[0]]
        return boxes,image,crp_img

    def preprocess_output(self, outputs):
        self.threshold = 0.5
        boxes = []
        probs = outputs[0, 0, :, 2]
        for i, p in enumerate(probs):
            if p > self.threshold:
                box = outputs[0, 0, i, 3:]
                boxes.append(box)
        return boxes
