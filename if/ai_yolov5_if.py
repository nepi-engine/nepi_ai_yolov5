#!/usr/bin/env python

import sys
import os
import os.path

import glob
import subprocess
import yaml
import time
import rospy
import numpy as np


from nepi_edge_sdk_base import nepi_ros


from std_msgs.msg import Empty, Float32
from nepi_ros_interfaces.msg import ObjectCount
from nepi_ros_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryResponse


from nepi_edge_sdk_base.save_cfg_if import SaveCfgIF


AI_NAME = 'Yolov5' # Use in display menus
FILE_TYPE = 'AIF'
AI_DICT = dict(
    description = 'Yolov5 ai framework support',
    pkg_name = 'nepi_ai_yolov5',
    class_name = 'Yolov5AIF',
    node_file = 'nepi_ai_yolov5_node.py',
    node_name = 'yolov5_ros',
    launch_file = 'yolov5_ros.launch',
    models_folder = 'yolov5_ros',
    model_prefix = 'yolov5_',
)

class Yolov5AIF(object):
    def __init__(self, ai_dict,node_base_namespace,models_lib_path):
      if node_base_namespace[-1] == "/":
        node_base_namespace = node_base_namespace[:-1]
      self.node_base_namespace = node_base_namespace
      self.models_lib_path = models_lib_path
      self.pkg_name = ai_dict['pkg_name']
      self.node_name = ai_dict['node_name']
      self.launch_file = ai_dict['launch_file']
      self.model_prefix = ai_dict['model_prefix']
      self.models_folder = ai_dict['models_folder']
      self.models_folder_path =  os.path.join(self.models_lib_path, self.models_folder)
      rospy.loginfo("Yolov5 models path: " + self.models_folder_path)
      threshold_namespace = self.node_base_namespace + '/' + self.node_name + '/set_threshold'
      self.set_threshold_pub = rospy.Publisher(threshold_namespace, Float32, queue_size=1, latch=True)
    
    #################
    # Yolov5 Model Functions

    def getModelsDict(self):
        models_dict = dict()
        classifier_name_list = []
        classifier_size_list = []
        classifier_classes_list = []
        # Try to obtain the path to Yolov5 models from the system_mgr
        cfg_path_config_folder = os.path.join(self.models_folder_path, 'configs')
        rospy.loginfo("Yolov5: Looking for models config files in folder: " + cfg_path_config_folder)
        # Grab the list of all existing yolov5 cfg files
        if os.path.exists(cfg_path_config_folder) == False:
            rospy.loginfo("Yolov5: Failed to find models config files in folder: " + cfg_path_config_folder)
            return models_dict
        else:
            self.cfg_files = glob.glob(os.path.join(cfg_path_config_folder,'*.yaml'))
            #rospy.loginfo("Yolov5: Found network config files: " + str(self.cfg_files))
            # Remove the ros.yaml file -- that one doesn't represent a selectable trained neural net
            for f in self.cfg_files:
                cfg_dict = dict()
                success = False
                try:
                    yaml_stream = open(f, 'r')
                    success = True
                except Exception as e:
                    rospy.logwarn("Yolov5: Failed to open yaml file: " + str(e))
                if success:
                    try:
                        # Validate that it is a proper config file and gather weights file size info for load-time estimates
                        cfg_dict = yaml.load(yaml_stream)  
                        classifier_keys = list(cfg_dict.keys())
                        classifier_key = classifier_keys[0]
                    except Exception as e:
                        rospy.logwarn("Yolov5: Failed load yaml data: " + str(e)) 
                        success = False 
                try: 
                    yaml_stream.close()
                except Exception as e:
                    rospy.logwarn("Yolov5: Failed close yaml file: " + str(e))
                
                if success == False:
                    rospy.logwarn("Yolov5: File does not appear to be a valid A/I model config file: " + f + "... not adding this classifier")
                    continue
                #rospy.logwarn("Yolov5: Import success: " + str(success) + " with cfg_dict " + str(cfg_dict))
                cfg_dict_keys = cfg_dict[classifier_key].keys()
                if ("cfg_file" not in cfg_dict_keys) or ("weight_file" not in cfg_dict_keys):
                    rospy.logwarn("Yolov5: File does not appear to be a valid A/I model config file: " + f + "... not adding this classifier")
                    continue


                classifier_name = os.path.splitext(os.path.basename(f))[0]
                weight_file = os.path.join(self.models_folder_path, "models", "weights",cfg_dict[classifier_key]["weight_file"]["name"])
                if not os.path.exists(weight_file):
                    rospy.logwarn("Yolov5: Classifier " + classifier_name + " specifies non-existent weights file " + weight_file + "... not adding this classifier")
                    continue
                classifier_classes_list.append(cfg_dict[classifier_key]['detection_classes']['names'])
                #rospy.logwarn("Yolov5: Classes: " + str(classifier_classes_list))
                classifier_name_list.append(classifier_name)
                classifier_size_list.append(os.path.getsize(weight_file))
            for i,name in enumerate(classifier_name_list):
                model_name = self.model_prefix + name
                model_dict = dict()
                model_dict['name'] = name
                model_dict['size'] = classifier_size_list[i]
                model_dict['classes'] = classifier_classes_list[i]
                models_dict[model_name] = model_dict
            #rospy.logwarn("Classifier returning models dict" + str(models_dict))
        return models_dict


    def startClassifier(self, classifier, source_img_topic, threshold):
        # Build Yolov5 new classifier launch command
        launch_cmd_line = [
            "roslaunch", self.pkg_name, self.launch_file,
            "pkg_name:=" + self.pkg_name,
            "node_base_namespace:=" + self.node_base_namespace, 
            "node_name:=" + self.node_name,
            "file_name:=" + self.file_name,
            "weights_path:=" + os.path.join(self.models_folder_path, "models/weights"),
            "config_path:=" + os.path.join(self.models_folder_path, "models/cfg"),
            "network_param_file:=" + os.path.join(self.models_folder_path, "configs", classifier + ".yaml"),
            "source_img_topic:=" + source_img_topic,
            "detector_threshold:=" + str(threshold)
        ]
        rospy.loginfo("Yolov5: Launching Yolov5 ROS Process: " + str(launch_cmd_line))
        self.ros_process = subprocess.Popen(launch_cmd_line)
        

        # Setup Classifier Setup Tracking Progress




    def stopClassifier(self):
        rospy.loginfo("Yolov5: Stopping classifier")
        if not (None == self.ros_process):
            self.ros_process.terminate()
            self.ros_process = None
        self.current_classifier = "None"
        self.current_img_topic = "None"
        
        #self.current_threshold = 0.3

    def updateClassifierThreshold(self,threshold):
        self.set_threshold_pub.publish(threshold)


    

if __name__ == '__main__':
    Yolov5AIF()
