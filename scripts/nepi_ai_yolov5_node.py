#!/usr/bin/env python

import os
import sys
import rospy
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
np.bool = np.bool_
import pandas

from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_msg
from nepi_edge_sdk_base import nepi_ais

from nepi_edge_sdk_base.ai_node_if import AiNodeIF

# Define your PyTorch model and load the weights
# model = ...

TEST_DETECTION_DICT_ENTRY = {
    'class_name': 'chair', # Class String Name
    'id': 1, # Class Index from Classes List
    'uid': '', # Reserved for unique tracking by downstream applications
    'probability': .3, # Probability of detection
    'box_xmin': 10,
    'box_ymin': 10,
    'box_xmax': 100,
    'box_ymax': 100
}



class PytorchDetector():

    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "ai_yolov5" # Can be overwitten by luanch command
    def __init__(self):
        #### APP NODE INIT SETUP ####
        nepi_ros.init_node(name= self.DEFAULT_NODE_NAME)
        self.node_name = nepi_ros.get_node_name()
        self.base_namespace = nepi_ros.get_base_namespace()
        self.node_namespace = self.base_namespace + self.node_name
        nepi_msg.createMsgPublishers(self)
        nepi_msg.publishMsgInfo(self,"Starting Initialization Processes")
        ##############################
        # Initialize parameters and fields.
        #node_params = nepi_ros.get_param(self,"~")
        #nepi_msg.publishMsgInfo(self,"Starting node params: " + str(node_params))
        self.model_name = nepi_ros.get_param(self,"~model_name","")
        self.pub_sub_namespace = nepi_ros.get_param(self,"~pub_sub_namespace",self.node_namespace)
        self.yolov5_path = nepi_ros.get_param(self,"~yolov5_path","")
        self.weights_path = nepi_ros.get_param(self,"~weights_path","")
        self.configs_path = nepi_ros.get_param(self,"~configs_path","")
        self.source_img_topic = nepi_ros.get_param(self,"~source_img_topic","")
        threshold_str = nepi_ros.get_param(self,"~detector_threshold","0.5")
        try:
            self.threshold = float(threshold_str)
        except Exception as e:
            self.threshold = 0.5
            nepi_msg.publishMsgWarn(self,"Failed to convert threshold str " + threshold_str + " to float")
        if self.model_name == "":
            nepi_msg.publishMsgWarn(self,"Failed to get required node info from param server: ")
            rospy.signal_shutdown("Failed to get valid model info from param")
        else:
            model_info = nepi_ros.get_param(self,"~ai_model","")
            if model_info == "":
                nepi_msg.publishMsgWarn(self,"Failed to get required model info from params: ")
                rospy.signal_shutdown("Failed to get valid model file paths")
            else:
                # Load the model
                # Add paths to python
                #nepi_msg.publishMsgWarn(self,"Got model info from param server: " + str(model_info))
                #self.appendProjectFolderPaths(self.yolov5_path)
                self.weight_file_path = os.path.join(self.weights_path, model_info['weight_file']['name'])
                self.config_file_path = os.path.join(self.configs_path, model_info['cfg_file']['name'])
                self.classes = model_info['detection_classes']['names']
                yolo_py_path = os.path.join(self.yolov5_path,'models')
                sys.path.append(yolo_py_path)
                # Load the model
                #YOLO = nepi_ais.importAIClass('yolo.py',yolo_py_path,'yolo','Model')
                #self.model = YOLO(self.weight_file_path)
                #self.load_state_dict(torch.load(self.weight_file_path))
                self.model = torch.hub.load(yolo_py_path,'custom', self.weight_file_path)
                self.model.eval()
   
                self.ai_if = AiNodeIF(node_name = self.node_name, 
                                    source_img_topic = self.source_img_topic,
                                    pub_sub_namespace = self.pub_sub_namespace,
                                    classes_list = self.classes,
                                    setThresholdFunction = self.setThreshold,
                                    processDetectionFunction = self.processDetection)

                #########################################################
                ## Initiation Complete
                nepi_msg.publishMsgInfo(self,"Initialization Complete")
                # Spin forever (until object is detected)
                nepi_ros.spin()
                #########################################################        
              

    def appendProjectFolderPaths(self,project_path):
        for entry in os.scandir(path):
            if entry.is_dir():
                rospy.logwarn(entry)

    def setThreshold(self,threshold):
        self.threshold = threshold
             

    def processDetection(self,cv2_img):
        # Convert BGR image to RGB image
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        # Define a transform to convert
        # the image to torch tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Convert the image to Torch tensor
        tensor = transform(image)
        # Update model settings
        self.model.conf = self.threshold  # Confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        self.model.max_det = 20  # Maximum number of detections per image
        # Run the detection model on tensor
        results = self.model(tensor)
        nepi_msg.publishMsgInfo(self,"Got Yolo detection results: " + str(results))   
        
        results.xyxy[0]  # img predictions (tensor)
        results_panda = results.pandas().xyxy[0]  # img1 predictions (pandas)
        nepi_msg.publishMsgInfo(self,"Got Panda formated Yolo detection results: " + str(results_panda))
        #      xmin    ymin    xmax   ymax  confidence  class    name
        # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
        # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
        # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
        # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
        

        return [TEST_DETECTION_DICT_ENTRY]



if __name__ == '__main__':
    PytorchDetector()
