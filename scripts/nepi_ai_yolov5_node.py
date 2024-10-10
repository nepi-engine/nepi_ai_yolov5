#!/usr/bin/env python

import os
import rospy
import torch

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
        self.weights_path = nepi_ros.get_param(self,"~weights_path","")
        self.config_path = nepi_ros.get_param(self,"~config_path","")
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
                #nepi_msg.publishMsgWarn(self,"Got model info from param server: " + str(model_info))
                self.weights_file_path = os.path.join(self.weights_path, model_info['weight_file']['name'])
                self.config_file_path = os.path.join(self.config_path, model_info['cfg_file']['name'])
                self.classes = model_info['detection_classes']['names']

                # Load the model
                # Initialize the YOLO model.
                '''
                # Update model settings
                self.model.conf = 0.3  # Confidence threshold (0-1)
                self.model.iou = 0.45  # NMS IoU threshold (0-1)
                self.model.max_det = 20  # Maximum number of detections per image
                self.model.eval()

                #  Convert image from ros to cv2
                cv2_img = nepi_img.rosimg_to_cv2img(source_img_msg)
                ros_timestamp = img_msg.header.stamp

                #  Run model against image
                results = self.model(self.img)
                '''


                '''
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = torch.load('/home/nepi/pytorch_ws/models/yolov5', \
                                            'custom', \
                                            source="local", \
                                            path="/home/nepi/pytorch_ws/models/yolov5/checkpoints/yolov5s.pt" \
                            ).to(self.device)
                
                '''
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
              

    def setThreshold(self,threshold):
        pass              

    def processDetection(self,cv2_img):
        return [TEST_DETECTION_DICT_ENTRY]





if __name__ == '__main__':
    PytorchDetector()
