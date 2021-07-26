# Computer Pointer Controller

In this project, we are trying to control the mouse pointer using various CNN models. Input for the model is either a webcam feed or the video provided by udacity. The project uses the following models,

1) Face detection - Helps to identify face/faces in the given video feed
2) Head pose detection - identifies the pose of the head with outputs Yaw,roll, pitch
3) Landmark estimation - identifies 5 key landmarks in teh faces, 2 eyes, 1 nose, 2 points in mouth. The eyes are cropped from this model
4) Gase estimation - The output of headpose and landmark are fed to gaze estimation, to obtain a vector. This vector directs t0 X,Y co-ordinates in screen

## Project Set Up and Installation

1) The current working directory is 'computer_pointer_controller'
2) All the models are downloaded and kept inside the folder 'intel'. On total there are 4 models present inside
3) Please install the libraries inside the requirement.txt file to run this model

## Demo
**step 1 :** Make sure you have the openvino setup installed in your local PC. Follow the link to install openvino : https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino
<br>
**step 2 :** Follow the above link and setup the environmental variable. After successfull linking of environmental variables, you should see "[setupvars.sh] OpenVINO environment initialized."
<br>
**step 3 :** Run " pip install -r requirements.txt"
<br>
**step 4 :**
!python inference.py --input_type video --face_path ./intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --landmark_path ./intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 --headpose_path ./intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 --gaze_path ./intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 --device CPU --inter_viz

## Documentation
The command line argument contains the following commands. You run !python inference.py --help, to understand the functions in command line arguments
1) input_type - whether to use webcam or video inside the folder 'demo.mp4'
2) face_path - path to face detection model
3) landmark_path - path to landmark detection model
4) headpose_path - path to headpose identification model
5) gaze_path - path to gaze estimation model
6) device - device to run the model (CPU,VPU,IGPU)
7) inter_viz - flag to visualize the video or intermediate outputs
 
## Benchmarks
model loading time, inference time on using the model in different devices like CPU, VPU, IGPU are shown in a seperate results.pdf file in the folder. Please check. Please check the load_time.jpg and inference_time.jpg images given in the folder. The time changes depending on teh number of supported layers and teh hetero plugin used.

## Results
There is a difference in loading and inference time. The model with FP32 takes more time than FP16 which takes more time than INT8. This is because, of the higher precision levels. FP32 has the highest precision level which inturn takes more space for storing the model. This increases its loading time and inference time. Both decreases as the precision decreases.


### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

Async inference basically helps to do a kind of parallel processing. It doesnt wait for the other image instead it takes and keeps processing as much as possible. 
This helps in reducing the latency and increasing the processing time.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

As the lighting decreases, the accuracy of the face detection model decreased. Some times the face was not detected. Normalizing the input image along with some basic image preprocessing helped to overcome the difficulty. 


