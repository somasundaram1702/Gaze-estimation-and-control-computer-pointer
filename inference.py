
import cv2
import logging
import numpy as np
import argparse
from face_detect import face_detect
from head_pose import head_pose
from landmark import landmark
from gaze import gaze
from openvino.inference_engine import IENetwork, IECore
from movemouse import MouseController


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def main(args):
    
        path_fd = args.face_path
        fd = face_detect('face detection',path_fd,args.device)
        fd.load_model()

        path_ld = args.landmark_path
        ld = landmark('landmark',path_ld,args.device)
        ld.load_model()

        path_hdps = args.headpose_path
        hp = head_pose('head pose',path_hdps,args.device)
        hp.load_model()

        gaze_path = args.gaze_path
        gz = gaze('Gaze',gaze_path,args.device)
        gz.load_model()

        if args.input_type == 'video':
            cap = cv2.VideoCapture('demo.mp4')
        elif args.input_type == 'cam':
            cap = cv2.VideoCapture(0)
            
        video_writer = cv2.VideoWriter('output1.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10,(1920,1080))


        if not cap.isOpened():
            logging.error('Video file not found. Check the path')


        while(cap.isOpened()):
            ret,frame = cap.read()
            if ret == True:
                #img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                boxes,pre_img,crp_img = fd.predict(frame)
                keypoint_image,right_eye,left_eye,x_e,y_e = ld.predict(crp_img)
                hp_vector = hp.predict(crp_img)
                hp_vector = np.reshape(hp_vector,(1,3))
                mouse_points = gz.predict(left_eye,right_eye,hp_vector)

                # rotation vector
                rvec = np.array([0, 0, 0], np.float)
                # translation vector
                tvec = np.array([0, 0, 0], np.float)
                # camera matrix
                camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float)

                result, _ = cv2.projectPoints(mouse_points, rvec, tvec, camera_matrix, None)
                result = result[0][0]

                res = (int(result[0] * 100), int(result[1] * 100))
                e1 = (boxes[0][0]+x_e[0],boxes[0][1]+y_e[0])
                e2 = (boxes[0][0]+x_e[1],boxes[0][1]+y_e[1])


                cv2.arrowedLine(pre_img, e1, (e1[0] - res[0], e1[1] + res[1]), (0, 255, 0), 2)
                cv2.arrowedLine(pre_img, e2, (e2[0] - res[0], e2[1] + res[1]), (0, 255, 0), 2)
                
                #move_mouse = MouseController('medium','medium')
                #move_mouse.move((e1[0] - res[0], e1[1] + res[1]))

                if (args.inter_viz):
                    cv2.imshow('frame',pre_img)
                    video_writer.write(frame)
                    cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type',default=None,
                   help = 'Enter \'video\' or \'image\' or \'cam\'')
    parser.add_argument('--face_path',default=None,
                   help = 'Enter the path for face detection model')
    parser.add_argument('--headpose_path',default=None,
                   help = 'Enter the path for head pose detection model')
    parser.add_argument('--landmark_path',default=None,
                   help = 'Enter the path for landmark detection model')
    parser.add_argument('--gaze_path',default=None,
                   help = 'Enter the path for gaze estimation model')
    parser.add_argument('--device',default='CPU',
                   help = 'Enter the device to run model')
    parser.add_argument('--inter_viz',action = 'store_true',
                   help = 'Flag for visualization')

    args=parser.parse_args()
    main(args)

