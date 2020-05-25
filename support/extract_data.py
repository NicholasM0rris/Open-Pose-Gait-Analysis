import os
from os import startfile
import sys
import glob
import json
from datetime import datetime
import cv2
import numpy as np
from collections import defaultdict
import argparse
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')

"""
Create argparse arguments
"""
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=False, help='Add image directory')
ap.add_argument('-v', '--video', required=False, help='Add Video')
ap.add_argument('-d', '--data', required=False, help='Add data directory')
ap.add_argument('-cd', '--cdata', required=False, help='Add coronal data directory')
ap.add_argument('-ci', '--cimages', required=False, help='Add coronal image directory')
ap.add_argument('-height', '--height', required=False, type=int, help='Add height of the person in centimetres (cm)')
ap.add_argument('-fps', '--fps', required=False, type=int, help='FPS to save video output')
ap.add_argument('-vl', '-video_length', required=False, type=float, help='Add the video length in seconds')

args = vars(ap.parse_args())

''' Initialise necessary directories  '''

try:
    if not os.path.exists('output_video'):
        os.makedirs('output_video')
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    if not os.path.exists('output_coronal_images'):
        os.makedirs('output_coronal_images')
    if not os.path.exists('processed_video'):
        os.makedirs('processed_video')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    if not os.path.exists('input_data'):
        os.makedirs('input_data')
    if not os.path.exists('input_images'):
        os.makedirs('input_images')
    if not os.path.exists('coronal_input_data'):
        os.makedirs('input_data')
    if not os.path.exists('coronal_input_images'):
        os.makedirs('input_images')

except OSError:
    print("OSERROR: Lacking perms to create directories")
    sys.exit()

# Define key point dictionary
key_points = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel": 24
}


class ExtractData:
    """
    Create a class for extracting data from JSON and image files
    """

    def __init__(self):
        self.cheese = 1
        if args['images']:
            self.path_to_input = args['images']
        else:
            self.path_to_input = "input_images"
        if args['cimages']:
            self.path_to_coronal_input = args['cimages']
        else:
            self.path_to_coronal_input = "coronal_input_images"

        self.video_dimensions = ""
        self.data_files = []
        self.input_files = []
        self.coronal_input_files = []
        self.coronal_data_files = []
        if args['data']:
            self.path = args['data']
        else:
            self.path = "input_data"
        if args['cdata']:
            self.coronal_path = args['cdata']
        else:
            self.coronal_path = "coronal_input_data"
        self.key_points = defaultdict(list)
        self.coronal_key_points = defaultdict(list)

        self.get_data_files()
        self.get_data_frames()
        self.extract_frames()
        self.interpolate_all_keypoints()
        self.interpolate_all_coronal_keypoints()

        if len(self.input_files) != len(self.data_files):
            print("The number of input files and data files differ! Closing program. Check they are the same"
                  "and try again!")
            sys.exit(1)
        if len(self.coronal_input_files) != len(self.coronal_data_files):
            print("The number of coronal input files and data files differ! Closing program. Check they are the same"
                  "and try again!")
            sys.exit(1)
        # print(len(self.input_files), len(self.data_files))
        # sys.exit()

    def print_keypoints(self, key_point=None):
        """
        Prints keypoint list
        :param key_point: Show specific key point
        """
        if key_point:
            print(self.key_points[key_point])
        else:
            print(self.key_points)

    def get_data_files(self):
        """
        Makes a list of keypoint files
        :return: list of keypoint files
        """
        for filename in glob.glob("{}\\*.JSON".format(self.path)):
            self.data_files.append(filename)
        for filename in glob.glob("{}\\*.JSON".format(self.coronal_path)):
            self.coronal_data_files.append(filename)

    def print_data_files(self):
        """
        Print current list of data files
        """
        print(self.data_files)

    def print_number_data_files(self, df):
        """
        Print number of data files
        :param df:
        :return:
        """
        self.cheese = 1
        print(len(df['people'][0]['pose_keypoints_2d']))

    def get_data_frames(self):
        """
        Extract the list of joint keypoint locations from JSON files stored in the list of paths in data_files
        :return:
        """
        try:
            # Get sagittal plane
            for files in self.data_files:
                temp = []
                temp_df = json.load(open(files))
                for key in key_points.keys():
                    self.key_points[key].append(
                        temp_df['people'][0]['pose_keypoints_2d'][key_points[key] * 3:key_points[key] * 3 + 3])

        # Except index error due to empty directory
        except IndexError:
            print("Error ! sagittal input folders may be empty !")
            sys.exit(1)
        try:
            # Get coronal plane
            for files in self.coronal_data_files:
                temp = []
                temp_df = json.load(open(files))
                for key in key_points.keys():
                    self.coronal_key_points[key].append(
                        temp_df['people'][0]['pose_keypoints_2d'][key_points[key] * 3:key_points[key] * 3 + 3])
        # Empty directories
        except IndexError:
            print("Error ! Coronal folders may be empty !")
            sys.exit(1)

    def extract_video_frame(self, sec=0):
        """
        Extract a video frame
        :param sec: Specify time to extract frame
        :return:
        """
        if args['video']:
            vidcap = cv2.VideoCapture(args['video'])

        else:
            vidcap = cv2.VideoCapture("output_video\\result.avi")

        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec)
        has_frames, frame = vidcap.read()
        if frame is None or not has_frames:
            print("Error reading file")
            sys.exit(1)
        return frame

    def extract_frames(self):
        """
        Extract the open pose image files for sagittal and coronal planes
        :return:
        """
        for filename in glob.glob("{}\\*.png".format(self.path_to_input)):
            self.input_files.append(filename)
        for filename in glob.glob("{}\\*.png".format(self.path_to_coronal_input)):
            self.coronal_input_files.append(filename)

    def interpolate_all_keypoints(self):
        """
        Interpolate between points to remove 0,0 points not found by Open Pose
        :return:
        """
        # print(self.key_points)
        # Get the key to change points
        for key in self.key_points.keys():
            # Iterate through all items in key and check for bad values
            for idx, item in enumerate(self.key_points[key]):
                # If bad value
                # print(idx, item, "TESTING", self.key_points[key], len(self.key_points[key])-1)
                if item[0] == 0 and item[1] == 0:
                    try:
                        # First frame - need to extrapolate instead
                        if idx == 0:
                            # If the frame after the first frame is not missing ( not zero )
                            if (tuple(self.key_points[key][idx + 1]) != (0, 0, 0)) and (tuple(self.key_points[key][idx + 2]) != (0, 0, 0)):
                                item[0] = 2 * self.key_points[key][idx + 1][0] - self.key_points[key][idx + 2][0]
                                item[1] = 2 * self.key_points[key][idx + 1][1] - self.key_points[key][idx + 2][1]
                            else:
                                # The points after the first frame are also missing. Find frame that is not missing
                                frame_x = self.key_points[key][idx + 1][0]
                                frame_y = self.key_points[key][idx + 1][1]
                                iterator = 0
                                try:
                                    while frame_x == 0 and frame_y == 0:
                                        iterator += 1
                                        frame_x = self.key_points[key][idx + 1 + iterator][0]
                                        frame_y = self.key_points[key][idx + 1 + iterator][1]
                                    # Assume the next frame will have valid data points and find difference
                                    diff_x = self.key_points[key][idx + 2 + iterator][0] - frame_x
                                    diff_y = self.key_points[key][idx + 2 + iterator][1] - frame_y
                                    # Correct all frames from first frame to valid data points
                                    for idx, i in enumerate(reversed(range(iterator+1))):
                                        self.key_points[key][i][0] -= diff_x * (idx + 1)
                                        self.key_points[key][i][1] -= diff_y * (idx + 1)
                                except IndexError:
                                    pass

                        # Last frame - need to extrapolate
                        elif idx == len(self.key_points[key]) - 1:
                            item[0] = 2 * self.key_points[key][idx - 1][0] - self.key_points[key][idx - 2][0]
                            item[1] = 2 * self.key_points[key][idx - 1][1] - self.key_points[key][idx - 2][1]
                        # Else interpolate average of f-1 and f+1
                        else:
                            # print("Changing index {} item {} to ".format(idx, item))
                            print('here', self.key_points[key][idx + 1])
                            # sys.exit()
                            if tuple(self.key_points[key][idx + 1]) != (0, 0, 0):

                                item[0] = (self.key_points[key][idx - 1][0] + self.key_points[key][idx + 1][0]) / 2
                                item[1] = (self.key_points[key][idx - 1][1] + self.key_points[key][idx + 1][1]) / 2
                            # print("this", item)

                    except IndexError as e:
                        print(key, self.key_points[key])
                        raise e
                        # print("Some index error in debugging . . . . . ")
                        # sys.exit()

    def interpolate_all_coronal_keypoints(self):
        """
        Interpolate between points to remove 0,0 points not found by Open Pose
        :return:
        """
        # print(self.key_points)
        # Get the key to change points
        for key in self.coronal_key_points.keys():
            # Iterate through all items in key and check for bad values
            for idx, item in enumerate(self.coronal_key_points[key]):
                # If bad value
                # print(idx, item, "TESTING", self.key_points[key], len(self.key_points[key])-1)
                if item[0] == 0 and item[1] == 0:
                    try:
                        # First frame - need to extrapolate instead
                        if idx == 0:
                            item[0] = 2 * self.coronal_key_points[key][idx + 1][0] - \
                                      self.coronal_key_points[key][idx + 2][0]
                            item[1] = 2 * self.coronal_key_points[key][idx + 1][1] - \
                                      self.coronal_key_points[key][idx + 2][1]
                        # Last frame - need to extrapolate
                        elif idx == len(self.coronal_key_points[key]) - 1:
                            item[0] = 2 * self.coronal_key_points[key][idx - 1][0] - \
                                      self.coronal_key_points[key][idx - 2][0]
                            item[1] = 2 * self.coronal_key_points[key][idx - 1][1] - \
                                      self.coronal_key_points[key][idx - 2][1]
                        # Else interpolate average of f-1 and f+1
                        else:
                            # print("Changing index {} item {} to ".format(idx, item))
                            try:
                                if self.coronal_key_points[key][idx + 1][0] == 0 and \
                                        self.coronal_key_points[key][idx + 1][1] == 0:
                                    item[0] = (self.coronal_key_points[key][idx - 1][0] +
                                               self.coronal_key_points[key][idx + 2][0]) / 2
                                    item[1] = (self.coronal_key_points[key][idx - 1][1] +
                                               self.coronal_key_points[key][idx + 2][1]) / 2
                                else:
                                    item[0] = (self.coronal_key_points[key][idx - 1][0] +
                                               self.coronal_key_points[key][idx + 1][0]) / 2
                                    item[1] = (self.coronal_key_points[key][idx - 1][1] +
                                               self.coronal_key_points[key][idx + 1][1]) / 2
                            except IndexError:
                                item[0] = (self.coronal_key_points[key][idx - 1][0] +
                                           self.coronal_key_points[key][idx + 1][0]) / 2
                                item[1] = (self.coronal_key_points[key][idx - 1][1] +
                                           self.coronal_key_points[key][idx + 1][1]) / 2
                            # print("this", item)
                    except IndexError:
                        print("Some index error in debugging . . . . . ")
                        sys.exit()


class DisplayData:
    """
    The class calculates and displays the processed data and measurements
    """

    def __init__(self, data):
        self.right_foot_count = 0
        self.left_foot_count = 0
        # Define list for index/frames in which step is made
        self.right_foot_index = []
        self.left_foot_index = []
        self.data = data
        # initial keypoints for distance
        self.keypoint1 = "LBigToe"
        self.keypoint2 = "RBigToe"
        self.gui = None
        # define distances
        self.distances = []
        self.distances.append("Distance from {} to {}".format(self.keypoint1, self.keypoint2))
        self.num_distances = []
        # Define angles made between two legs at the waist
        self.angles = []
        self.num_angles = []
        # Define angles made between legs and torso
        self.leg_body_angles = []
        self.num_leg_body_angles = []
        # define angles made at knee joint
        self.right_knee_angles = []
        self.num_right_knee_angles = []
        self.left_knee_angles = []
        self.num_left_knee_angles = []
        # Define step width between heels
        self.step_width = []
        self.num_step_width = []
        # Define angles between heel, big toe and the torso allignment
        self.foot_angles = []
        self.num_foot_angles = []
        # Frame lists
        self.frame_list = []
        self.coronal_frame_list = []
        self.frame_number = 1
        # Get video analytics
        if args['video']:
            self.video_path = args['video']
        else:
            self.video_path = 'op_video/test2.avi'
        self.duration, self.frame_count, self.fps = get_video_length(self.video_path)
        # Get number of steps
        self.velocity_list = []
        self.stride_length_list = []
        self.cadence = []
        self.correct_leg_swap()
        self.get_number_steps()
        self.get_velocity()
        self.get_cadence()

    def plot_points(self, keypoint):
        """
        Plot the points of a keypoint for each frame
        :param keypoint: Keypoint to plot e.g "RHeel"
        :return:
        """
        frame = cv2.imread(self.data.input_files[0])
        height, width, layers = frame.shape
        x = []
        y = []
        area = 10
        colors = (0, 0, 0)
        for idx, point in enumerate(self.data.key_points):
            points = self.fp(keypoint, idx)
            x.append(points[0])
            y.append(points[1])
        print(x)
        print(y)
        print(self.data.key_points["RBigToe"])
        plt.scatter(x, y, c=colors, s=area, alpha=0.5)
        plt.title('Scatter plot')
        axes = plt.gca()
        axes.set_xlim([0, width])
        axes.set_ylim([0, height])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().invert_yaxis()
        plt.savefig("plots/scatter.png")

    def save_text(self, alist, text_type):
        """
        Saves a test file containing the data log
        :return:
        """

        time_stamp = datetime.now()
        filename = str("metrics/{}_log_{}.txt".format(text_type, time_stamp.strftime("%Y-%m-%d_%H-%M-%S")))
        f = open(filename, "w+")
        for something in alist:
            f.write("%s\n" % something)
        f.close()
        self.frame_number = 1

    # What does fp actually stand for? I forgot a long time ago...
    # fp stands for fast point to quickly retreive a keypoint coordinate
    def fp(self, keypoint, frame_index):
        """
        e.g fp("RBigToe, 1) will get x,y coord of RBigToe from frame 1
        Returns keypoint as x,y coordinate corresponding to index
        :param keypoint: string that is key of dictionary e.g "RBigToe"
        :param frame_index: what frame to access
        :return:
        """
        return self.data.key_points[keypoint][frame_index][:-1]

    def fp2(self, keypoint, frame_index):
        """
        e.g fp("RBigToe, 1) will get x,y coord of RBigToe from frame 1
        Returns keypoint as x,y coordinate corresponding to index
        :param keypoint: string that is key of dictionary e.g "RBigToe"
        :param frame_index: what frame to access
        :return:
        """
        return self.data.coronal_key_points[keypoint][frame_index][:-1]

    def add_points_to_image(self, frame, keypoints):
        """
        Overlay points to an image
        :param frame: Image/frame for points to be overlayed (from extract_frame) in red
        :param keypoints: list of keypoint coordinates to overlay
        :return: writes frame to output_images
        """
        for keypoint in keypoints:
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 10, (0, 0, 255), -1)
        return frame

    def add_line_between_points(self, frame, points, thickness):
        """
        Adds a line overlay between two points and puts pixel distance text
        :param frame:
        :param points:
        :return:
        """
        point1 = list(map(int, points[0]))
        point2 = list(map(int, points[1]))
        cv2.line(frame, tuple(point1), tuple(point2), (0, 255, 0), thickness=thickness, lineType=8)

        # print("pt1,pt2", point1, point2)

        return frame

    def distance_overlay(self):
        """
        Creates overlay for distance over the image frames
        :return:
        """
        # Add overlay
        if not self.frame_list:
            for idx, path in enumerate(self.data.input_files):
                frame = cv2.imread(path)
                frame = self.add_points_to_image(frame, [self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx)], 3)
                org = tuple([int(self.fp(self.keypoint1, idx)[0]), int(self.fp(self.keypoint1, idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                dist = get_distance(self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx))
                self.distances.append("Frame {} - Distance: {}".format(self.frame_number, dist))
                self.num_distances.append(dist)
                frame = cv2.putText(frame, 'Distance: {}'.format(dist), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                self.frame_list.append(frame)
                # save_frame(frame)
        # If there has already been frames processed (frame_list not empty)
        else:
            temp_list = []
            for idx, frame in enumerate(self.frame_list):
                frame = self.add_points_to_image(frame, [self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx)], 3)
                org = tuple([int(self.fp(self.keypoint1, idx)[0]), int(self.fp(self.keypoint1, idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                dist = get_distance(self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx))
                self.distances.append("Frame {} - Distance: {}".format(self.frame_number, dist))
                self.num_distances.append(dist)
                frame = cv2.putText(frame, 'Distance: {}'.format(dist), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                temp_list.append(frame)
            self.frame_list = temp_list
        if self.gui.distance_checkbox == Qt.Checked:
            print("Saving distances to text file")
            self.save_text(self.distances, "Distance")
        max_dist = 0
        min_dist = 999
        # Get max and min distances
        for dist in self.num_distances:
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist
        self.gui.max_dist_label.setText("Max dist: {}".format(max_dist))
        self.gui.min_dist_label.setText("Min dist: {}".format(min_dist))

    def get_angle(self, p3, p2, p1):
        """
        a = (p1.x - p2.x, p1.y - p2.y)
        b = (p1.x - p3.x, p1.y - p3.y)
        a dot b = mag(a)mag(b)cos(theta)
        Returns the angle from three points by finding the cross product
        :param p3: point 1 (x,y)
        :param p2: point 2 (x,y)
        :param p1: common point to pt 1 & 2 (x,y)
        :return: angle in degrees
        """
        a = (p1[0] - p2[0], p1[1] - p2[1])
        b = (p1[0] - p3[0], p1[1] - p3[1])
        numerator = np.dot(a, b)
        denominator = (get_mag(a) * get_mag(b))
        angle = np.arccos(numerator / denominator)
        print("ANGLE", angle)

        # Convert to degrees
        angle = np.degrees(angle)

        return angle

    def get_left_vector_angle(self, idx):
        """
        180 - abs(arctan(m1) - arctan(m2))
        Get the angle between two lines:
        Line made from heel to toe, and from MidHip to Neck
        :return:
        """
        # Get gradients
        pt1 = self.fp2("LHeel", idx)
        pt2 = self.fp2("LBigToe", idx)
        try:
            m1 = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
        except ZeroDivisionError:
            m1 = 99999999
        pt3 = self.fp2("MidHip", idx)
        pt4 = self.fp2("Neck", idx)
        print([pt1, pt2, pt3, pt4])
        try:
            m2 = (pt3[1] - pt4[1]) / (pt3[0] - pt4[0])
        except ZeroDivisionError:
            m2 = 99999999
        angle = np.pi - abs(np.arctan(m1) - np.arctan(m2))
        return np.degrees(angle)

    def get_right_vector_angle(self, idx):
        """
        180 - abs(arctan(m1) - arctan(m2))
        Get the angle between two lines:
        Line made from heel to toe, and from MidHip to Neck
        :return:
        """
        # Get gradients
        pt1 = self.fp2("RHeel", idx)
        pt2 = self.fp2("RBigToe", idx)
        try:
            m1 = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
        except ZeroDivisionError:
            m1 = 99999999
        pt3 = self.fp2("MidHip", idx)
        pt4 = self.fp2("Neck", idx)
        try:
            m2 = (pt3[1] - pt4[1]) / (pt3[0] - pt4[0])
        except ZeroDivisionError:
            m2 = 99999999
        angle = np.pi - abs(np.arctan(m1) - np.arctan(m2))
        return np.degrees(angle)

    def angle_overlay(self):
        """
        Creates overlay for angle between legs from the hips
        (Such that there are two vectors from left knee to hips and right knee to hips
        :return:
        """
        # Add overlay
        if not self.frame_list:
            for idx, path in enumerate(self.data.input_files):
                frame = cv2.imread(path)
                frame = self.add_points_to_image(frame,
                                                 [self.fp("RKnee", idx), self.fp("LKnee", idx), self.fp("MidHip", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp("RKnee", idx), self.fp("MidHip", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp("LKnee", idx), self.fp("MidHip", idx)], 16)
                org = tuple([int(self.fp("MidHip", idx)[0]), int(self.fp("MidHip", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                angle = self.get_angle(self.fp("LKnee", idx), self.fp("RKnee", idx), self.fp("MidHip", idx))
                self.angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_angles.append(angle)
                frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                # save_frame(frame)
                self.frame_list.append(frame)
        else:
            """ If frame list is not empty, use the existing frames """
            temp_list = []
            for idx, frame in enumerate(self.frame_list):
                frame = self.add_points_to_image(frame,
                                                 [self.fp("RKnee", idx), self.fp("LKnee", idx), self.fp("MidHip", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp("RKnee", idx), self.fp("MidHip", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp("LKnee", idx), self.fp("MidHip", idx)], 16)
                org = tuple([int(self.fp("MidHip", idx)[0]), int(self.fp("MidHip", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                angle = self.get_angle(self.fp("LKnee", idx), self.fp("RKnee", idx), self.fp("MidHip", idx))
                self.angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_angles.append(angle)
                frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                temp_list.append(frame)

            self.frame_list = temp_list

        """ Save the calculated angles to a text file """
        if self.gui.angle_checkbox == Qt.Checked:
            print("Saving angles to text file")
            self.save_text(self.angles, "Angle")
        max_angle = 0
        for an_angle in self.num_angles:
            if an_angle > max_angle:
                max_angle = an_angle

        self.gui.angle_label.setText("Max Angle: {}".format(max_angle))

    def leg_body_angle_overlay(self):
        """
        Creates overlay for angle between legs and torso from the hips
        :return:
        """
        # Add overlay
        if not self.frame_list:
            for idx, path in enumerate(self.data.input_files):
                frame = cv2.imread(path)
                frame = self.add_points_to_image(frame,
                                                 [self.fp("LKnee", idx), self.fp("Neck", idx), self.fp("MidHip", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp("LKnee", idx), self.fp("MidHip", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp("Neck", idx), self.fp("MidHip", idx)], 16)
                org = tuple([int(self.fp("MidHip", idx)[0]), int(self.fp("MidHip", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                angle = self.get_angle(self.fp("Neck", idx), self.fp("LKnee", idx), self.fp("MidHip", idx))
                self.leg_body_angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_leg_body_angles.append(angle)
                frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                # save_frame(frame)
                self.frame_list.append(frame)

        else:
            """ If frame list is not empty, use the existing frames """
            temp_list = []
            for idx, frame in enumerate(self.frame_list):
                frame = self.add_points_to_image(frame,
                                                 [self.fp("LKnee", idx), self.fp("Neck", idx), self.fp("MidHip", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp("LKnee", idx), self.fp("MidHip", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp("Neck", idx), self.fp("MidHip", idx)], 16)
                org = tuple([int(self.fp("MidHip", idx)[0]), int(self.fp("MidHip", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                angle = self.get_angle(self.fp("Neck", idx), self.fp("LKnee", idx), self.fp("MidHip", idx))
                self.leg_body_angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_leg_body_angles.append(angle)
                frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                temp_list.append(frame)

            self.frame_list = temp_list

        if self.gui.leg_angle_body_checkbox == Qt.Checked:
            print("Saving angles to text file")
            self.save_text(self.leg_body_angles, "Angle")

    def right_knee_angle_overlay(self):
        """
        Creates overlay for angle at the right knee
        :return:
        """
        # Add overlay
        if not self.frame_list:
            for idx, path in enumerate(self.data.input_files):
                frame = cv2.imread(path)
                frame = self.add_points_to_image(frame,
                                                 [self.fp("RKnee", idx), self.fp("RHeel", idx), self.fp("RHip", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp("RKnee", idx), self.fp("MidHip", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp("RHeel", idx), self.fp("RKnee", idx)], 16)
                org = tuple([int(self.fp("RKnee", idx)[0]), int(self.fp("RKnee", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                angle = self.get_angle(self.fp("RHeel", idx), self.fp("RHip", idx), self.fp("RKnee", idx))
                self.right_knee_angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_right_knee_angles.append(angle)
                frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                # save_frame(frame)
                self.frame_list.append(frame)
        else:
            """ If frame list is not empty, use the existing frames """
            temp_list = []
            for idx, frame in enumerate(self.frame_list):
                frame = self.add_points_to_image(frame,
                                                 [self.fp("RKnee", idx), self.fp("RHeel", idx), self.fp("RHip", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp("RKnee", idx), self.fp("RHip", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp("RHeel", idx), self.fp("RKnee", idx)], 16)
                org = tuple([int(self.fp("RKnee", idx)[0]), int(self.fp("RKnee", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                angle = self.get_angle(self.fp("RHeel", idx), self.fp("RHip", idx), self.fp("RKnee", idx))
                self.right_knee_angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_right_knee_angles.append(angle)
                frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                temp_list.append(frame)

            self.frame_list = temp_list

        if self.gui.right_knee_angle_checkbox == Qt.Checked:
            print("Saving angles to text file")
            self.save_text(self.right_knee_angles, "Angle")

    def left_knee_angle_overlay(self):
        """
        Creates overlay for angle at the left knee
        :return:
        """
        # Add overlay
        if not self.frame_list:
            for idx, path in enumerate(self.data.input_files):
                frame = cv2.imread(path)
                frame = self.add_points_to_image(frame,
                                                 [self.fp("LKnee", idx), self.fp("LHeel", idx), self.fp("LHip", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp("LKnee", idx), self.fp("LHip", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp("LHeel", idx), self.fp("LKnee", idx)], 16)
                org = tuple([int(self.fp("LKnee", idx)[0]), int(self.fp("LKnee", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                angle = self.get_angle(self.fp("LHeel", idx), self.fp("LHip", idx), self.fp("LKnee", idx))
                self.left_knee_angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_left_knee_angles.append(angle)
                frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                # save_frame(frame)
                self.frame_list.append(frame)
        else:
            """ If frame list is not empty, use the existing frames """
            temp_list = []
            for idx, frame in enumerate(self.frame_list):
                frame = self.add_points_to_image(frame,
                                                 [self.fp("LKnee", idx), self.fp("LHeel", idx), self.fp("LHip", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp("LKnee", idx), self.fp("LHip", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp("LHeel", idx), self.fp("LKnee", idx)], 16)
                org = tuple([int(self.fp("LKnee", idx)[0]), int(self.fp("LKnee", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                """ Calculate the angle and save it """
                angle = self.get_angle(self.fp("LHeel", idx), self.fp("LHip", idx), self.fp("LKnee", idx))
                self.left_knee_angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_left_knee_angles.append(angle)
                frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                temp_list.append(frame)

            self.frame_list = temp_list

        if self.gui.left_knee_angle_checkbox == Qt.Checked:
            print("Saving angles to text file")
            self.save_text(self.left_knee_angles, "Angle")

    def get_step_width(self, index):
        """
        Get the coronal plane step width distances
        :param index: The frame index to retreive the points from
        :return:
        """
        return abs(self.fp2("RHeel", index)[0] - self.fp2("LHeel", index)[0])

    def display_step_width(self):
        """
        Display the step width in the coronal plane, calculated as the distance between left and right heel
        :return:
        """
        # Add overlay
        # print("frame list : ", self.data.coronal_input_files)
        # print("frame files : ", self.data.coronal_data_files)
        # print("frame files : ", self.data.input_files)
        if not self.coronal_frame_list:
            for idx, path in enumerate(self.data.coronal_input_files):
                frame = cv2.imread(path)
                frame = self.add_points_to_image(frame, [self.fp2("RHeel", idx), self.fp2("LHeel", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp2("RHeel", idx),
                                                      (self.fp2("LHeel", idx)[0], self.fp2("RHeel", idx)[1])], 3)
                org = tuple([int(self.fp2("RHeel", idx)[0]), int(self.fp2("LHeel", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                dist = self.get_step_width(idx)
                self.step_width.append("Frame {} - Step width: {}".format(self.frame_number, dist))
                self.num_step_width.append(dist)
                frame = cv2.putText(frame, 'Step width: {}'.format(dist), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                self.coronal_frame_list.append(frame)
                # save_frame(frame)
        # If there has already been frames processed (frame_list not empty)
        else:
            """ If frame list is not empty, use the existing frames """
            temp_list = []
            for idx, frame in enumerate(self.coronal_frame_list):
                frame = self.add_points_to_image(frame, [self.fp2("RHeel", idx), self.fp2("LHeel", idx)])
                frame = self.add_line_between_points(frame,
                                                     [self.fp2("RHeel", idx), self.fp2("LHeel", idx)], 3)
                org = tuple([int(self.fp2("RHeel", idx)[0]), int(self.fp2("LHeel", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                dist = self.get_step_width(idx)
                self.step_width.append("Frame {} - Step width: {}".format(self.frame_number, dist))
                self.num_step_width.append(dist)
                frame = cv2.putText(frame, 'Step width: {}'.format(dist), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                temp_list.append(frame)
            self.coronal_frame_list = temp_list
        if self.gui.coronal_checkbox == Qt.Checked:
            print("Saving step width to text file")
            self.save_text(self.step_width, "Step width")
        max_dist = 0
        min_dist = 999
        # Get max and min distances
        sum = 0
        for dist in self.num_step_width:
            sum += dist
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist
        # self.gui.max_dist_label.setText("Max dist: {}".format(max_dist))
        # self.gui.min_dist_label.setText("Min dist: {}".format(min_dist))
        print(min_dist, max_dist)
        self.gui.average_step_width_label.setText("Average step width: {}".format(sum / len(self.num_step_width)))

    def foot_angle_overlay(self):
        """
        Creates overlay for angle between foot. Calculated as angle between two vectors made from toe and heel, and
        Midhip to neck
        :return:
        """
        # Add overlay
        if not self.coronal_frame_list:
            for idx, path in enumerate(self.data.coronal_input_files):
                frame = cv2.imread(path)

                frame = self.add_points_to_image(frame,
                                                 [self.fp2("RHeel", idx), self.fp2("LHeel", idx),
                                                  self.fp2("RBigToe", idx), self.fp2("LBigToe", idx)])

                frame = self.add_line_between_points(frame,
                                                     [self.fp2("LHeel", idx), self.fp2("LBigToe", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp2("RHeel", idx), self.fp2("RBigToe", idx)], 16)
                org = tuple([int(self.fp2("RBigToe", idx)[0]), int(self.fp2("RBigToe", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                left_angle = self.get_left_vector_angle(idx)
                right_angle = self.get_right_vector_angle(idx)
                self.foot_angles.append("Frame {} - Left foot angle: {}".format(self.frame_number, left_angle))
                self.foot_angles.append("Frame {} - Right foot angle: {}".format(self.frame_number, right_angle))
                self.num_angles.append([left_angle, right_angle])
                frame = cv2.putText(frame, 'Right angle: {}'.format(right_angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                # save_frame(frame)
                self.coronal_frame_list.append(frame)
        else:
            """ If frame list is not empty, use the existing frames """
            temp_list = []
            for idx, frame in enumerate(self.coronal_frame_list):
                frame = self.add_points_to_image(frame,
                                                 [self.fp2("RHeel", idx), self.fp2("LHeel", idx),
                                                  self.fp2("RBigToe", idx), self.fp2("LBigToe", idx)])

                frame = self.add_line_between_points(frame,
                                                     [self.fp2("LHeel", idx), self.fp2("LBigToe", idx)], 16)
                frame = self.add_line_between_points(frame,
                                                     [self.fp2("RHeel", idx), self.fp2("RBigToe", idx)], 16)
                org = tuple([int(self.fp2("RBigToe", idx)[0]), int(self.fp2("RBigToe", idx)[1])])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (0, 0, 255)
                thickness = 2
                left_angle = self.get_left_vector_angle(idx)
                right_angle = self.get_right_vector_angle(idx)
                self.foot_angles.append("Frame {} - Left foot angle: {}".format(self.frame_number, left_angle))
                self.foot_angles.append("Frame {} - Right foot angle: {}".format(self.frame_number, right_angle))
                self.num_angles.append([left_angle, right_angle])
                frame = cv2.putText(frame, 'Right angle: {}'.format(right_angle), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                temp_list.append(frame)

            self.coronal_frame_list = temp_list

        if self.gui.foot_angle_checkbox == Qt.Checked:
            print("Saving foot angles to text file")
            self.save_text(self.foot_angles, "Foot Angle")
        max_angle = 0
        sum = 0
        for an_angle in self.num_angles:
            sum += an_angle[1]
            if an_angle[1] > max_angle:
                max_angle = an_angle[1]

        # self.gui.angle_label.setText("Max Angle: {}".format(max_angle))

    def correct_leg_swap(self):
        """
        Search through the datapoints to find cases where left and right leg have swapped
        Check the acceleration of the knee, and if detected then
        xHeel, xBigToe, xKnee, xSmallToe and xAnkle should be swapped
        :return:
        """
        right_knee_direction = 0
        left_knee_direction = 0
        ''' Define initial x1 and x2 '''
        RKnee_x1 = self.fp("RKnee", 0)[0]
        RKnee_x2 = self.fp("RKnee", 1)[0]
        RKnee_x3 = self.fp("RKnee", 2)[0]
        if RKnee_x1 > RKnee_x2:
            right_knee_direction = 1
        elif RKnee_x1 < RKnee_x2:
            right_knee_direction = 2
        ''' Get the rate of change '''
        change1 = RKnee_x2 - RKnee_x1
        change2 = RKnee_x3 - RKnee_x2
        rate_change = [change1 - change2]
        average_rate_change = 0
        ''' Iterate over the data files to look for leg swaps '''
        for idx, path in enumerate(self.data.input_files):
            try:
                RKnee_x1 = self.fp("RKnee", idx)[0]
                RKnee_x2 = self.fp("RKnee", idx + 1)[0]
                RKnee_x3 = self.fp("RKnee", idx + 2)[0]
                change1 = RKnee_x2 - RKnee_x1
                change2 = RKnee_x3 - RKnee_x2
                rate_change.append(change1 - change2)

            except IndexError:
                pass

        abs_sum = 0
        ''' Sum the absolute values of the rate changes'''
        for value in rate_change:
            abs_sum += abs(value)
        average_rate_change = abs_sum / len(rate_change)
        ''' Check if there is a rate change twice the average rate change'''
        detected_frame_list = []
        for idx, frame in enumerate(self.data.input_files):
            try:
                ''' If a high rate change is detected swap the legs'''
                if rate_change[idx] > average_rate_change * 3:
                    detected_frame_list.append(idx)
                    print("Possible leg swap at frame {}".format(idx))

                    ''' Get the points '''
                    RKnee = self.fp("RKnee", idx)
                    RAnkle = self.fp("RAnkle", idx)
                    RBigToe = self.fp("RBigToe", idx)
                    RSmallToe = self.fp("RSmallToe", idx)
                    RHeel = self.fp("RHeel", idx)

                    LKnee = self.fp("LKnee", idx)
                    LAnkle = self.fp("LAnkle", idx)
                    LBigToe = self.fp("LBigToe", idx)
                    LSmallToe = self.fp("LSmallToe", idx)
                    LHeel = self.fp("LHeel", idx)

                    ''' Do the swapping '''
                    LKnee, RKnee = RKnee, LKnee
                    LAnkle, RAnkle = RAnkle, LAnkle
                    LBigToe, RBigToe = RBigToe, LBigToe
                    LSmallToe, RSmallToe = RSmallToe, LSmallToe
                    LHeel, RHeel = RHeel, LHeel

            except IndexError:
                pass
        self.save_text(detected_frame_list, "Detected_leg_swap_frame_list")

    def get_number_steps(self):
        """
        Calculate the number of steps from the sagittal plane
        Direction 1: left
        direction 2: right
        :return:
        """
        right_direction = 0
        left_direction = 0
        x1_r = self.fp("RHeel", 0)[0]
        x2_r = self.fp("RHeel", 1)[0]
        if x1_r > x2_r:
            right_direction = 1
        elif x1_r < x2_r:
            right_direction = 2

        y1_r = self.fp("RHeel", 0)[1]
        y2_r = self.fp("RHeel", 1)[1]

        x1_left = self.fp("LHeel", 0)[0]
        x2_left = self.fp("LHeel", 1)[0]
        if x1_left > x2_left:
            left_direction = 1
        elif x1_left < x2_left:
            left_direction = 2
        print("init left direction", left_direction, x1_left, x2_left)

        y1_left = self.fp("LHeel", 0)[1]
        y2_left = self.fp("LHeel", 1)[1]

        for idx, path in enumerate(self.data.input_files):
            try:
                x1_r = self.fp("RHeel", idx)[0]
                x2_r = self.fp("RHeel", idx + 1)[0]
                if x1_r > x2_r:
                    if right_direction == 2:
                        # If moving right, and now moving left
                        y1_r = self.fp("RHeel", idx)[1]
                        y2_r = self.fp("RHeel", idx + 1)[1]
                        initial_rate = abs(y1_r - y2_r)
                        try:
                            for i in range(5):
                                y1_r = self.fp("RHeel", idx + i)[1]
                                y2_r = self.fp("RHeel", idx + 1 + i)[1]
                                rate = abs(y1_r - y2_r)
                                i += 1
                                if rate < initial_rate / 2:
                                    self.right_foot_count += 1
                                    self.right_foot_index.append(idx + i)
                                    break

                        except IndexError:
                            pass

                        # self.right_foot_count += 1
                        # self.right_foot_index.append(idx)
                    # Set direction to left
                    right_direction = 1

                elif x1_r < x2_r:
                    if right_direction == 1:
                        # If moving left, and now moving right
                        pass  # self.right_foot_count += 1
                    right_direction = 2
                else:
                    right_direction = 0
                print(x1_r, x2_r, idx, right_direction, self.right_foot_count)

                ##############################################
                x1_left = self.fp("LHeel", idx)[0]
                x2_left = self.fp("LHeel", idx + 1)[0]
                """
                if idx == 8:
                    print(left_direction, x1_left, x2_left)
                    sys.exit()
                """
                if x1_left > x2_left:
                    if left_direction == 2:
                        # If moving right, and now moving left
                        y1_left = self.fp("LHeel", idx)[1]
                        y2_left = self.fp("LHeel", idx + 1)[1]
                        initial_rate = abs(y1_left - y2_left)
                        try:
                            for i in range(5):
                                y1_left = self.fp("LHeel", idx + i)[1]
                                y2_left = self.fp("LHeel", idx + 1 + i)[1]
                                rate = abs(y1_left - y2_left)
                                i += 1
                                if rate < initial_rate / 2:
                                    self.left_foot_count += 1
                                    self.left_foot_index.append(idx + i)
                                    break
                        except IndexError:
                            pass

                    # Set direction to left
                    left_direction = 1

                elif x1_left < x2_left:
                    if left_direction == 1:
                        # If moving left, and now moving right
                        pass  # self.left_foot_count += 1

                    left_direction = 2

            except IndexError:
                print(self.left_foot_count, self.right_foot_count)
                print(self.left_foot_index)
                pass
        # print(self.left_foot_count, right_foot_count)
        # print(self.data.input_files)

    def display_step_number_overlay(self):

        """
        Creates overlay for step number over all frames
        :return:
        """
        org = (100, 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        color = (0, 0, 255)
        thickness = 2
        left_foot_count = 0
        right_foot_count = 0
        # Add overlay
        if not self.frame_list:
            for idx, path in enumerate(self.data.input_files):
                if idx in self.left_foot_index:
                    left_foot_count += 1
                if idx in self.right_foot_index:
                    right_foot_count += 1
                frame = cv2.imread(path)
                frame = cv2.putText(frame, 'Number of right steps: {}'.format(right_foot_count), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                frame = cv2.putText(frame, 'Number of left steps: {}'.format(left_foot_count), (100, 50), font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                frame = cv2.putText(frame, 'Frame count: {}'.format(idx), (100, 150), font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                self.frame_list.append(frame)
        else:
            temp_list = []

            for idx, frame in enumerate(self.frame_list):
                if idx in self.left_foot_index:
                    left_foot_count += 1
                if idx in self.right_foot_index:
                    right_foot_count += 1
                frame = cv2.putText(frame, 'Number of right steps: {}'.format(right_foot_count), org, font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                frame = cv2.putText(frame, 'Number of left steps: {}'.format(left_foot_count), (100, 50), font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                frame = cv2.putText(frame, 'Frame count: {}'.format(idx), (100, 150), font,
                                    fontscale, color, thickness, cv2.LINE_AA)
                self.frame_number += 1
                temp_list.append(frame)

            self.frame_list = temp_list

    def get_velocity(self):
        """
        Calculate the displacement between each step, and use it to find the velocity
        :return:
        """

        ''' Calculate velocity
            speed = displacement/time
            |x2 - x1| / time
            time = 1 / FPS
        '''
        area = 10
        colors = (0, 0, 0)
        save_to_file_list = []
        step_list = self.right_foot_index + self.left_foot_index
        step_list.sort()
        for x, idx in enumerate(step_list):
            try:
                displacement = abs(get_distance(self.fp("LBigToe", idx), self.fp("RBigToe", idx)))
                self.stride_length_list.append(
                    "Stride length: {}. Frame: {}. Step count: {}".format(displacement, idx, x))
                number_frames_passed = idx - step_list[x - 1]
                t = number_frames_passed * (1 / self.fps)
                speed = displacement / t
                if speed == float('+inf') or speed == float('-inf'):
                    speed = 999

                self.velocity_list.append(speed)
                save_to_file_list.append("The velocity at frame {}: {}".format(idx, speed))

            except IndexError:
                pass
        # Save the stride length to file
        self.save_text(self.stride_length_list, "Stride_length")
        filtered_list = np.array(self.velocity_list)
        mean = np.mean(filtered_list)
        std = np.std(filtered_list)
        print("Average velocity: ", mean)
        print("std: ", std)
        save_to_file_list.append("Unfiltered average velocity: {}".format(mean))
        save_to_file_list.append("Unfiltered standard deviation: {}".format(std))

        # Plot unfiltered

        plt.figure()
        plt.plot(step_list, self.velocity_list, linewidth=2, linestyle="-", c="b")
        plt.title('Unfiltered velocities')
        axes = plt.gca()
        ymin = 0
        ymax = 5000
        axes.set_ylim([ymin, ymax])
        plt.xlabel('frame number')
        plt.ylabel('velocities')
        # axes.set_yscale('log')
        # plt.gca().invert_yaxis()
        plt.savefig("plots/Unfiltered_velocities_scatter3.png")

        ''' Filter points for outliers '''
        idx_list = []
        # Get the indexes to remove
        data = np.array(self.velocity_list)
        d = np.abs(data - np.median(data))
        # Get the median and absolute distance to median
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        m = 2
        # print(data[s<2])

        ''' Get the indexes of the outliers'''
        for idx, x in enumerate(self.velocity_list):
            if s[idx] > m:
                idx_list.append(idx)

        # remove the indexes of outliers
        for index in sorted(idx_list, reverse=True):
            del self.velocity_list[index]
            del step_list[index]

        print(self.velocity_list)
        # filtered_list = [x for x in filtered_list if (x > mean - 2 * std)]
        # filtered_list = [x for x in filtered_list if (x < mean + 2 * std)]
        mean = np.mean(self.velocity_list)
        std = np.std(self.velocity_list)
        save_to_file_list.append("Filtered average velocity: {}".format(mean))
        save_to_file_list.append("Filtered standard deviation: {}".format(std))
        self.save_text(save_to_file_list, "Velocities")
        # self.velocity_list = filtered_list
        plt.figure()
        plt.scatter(step_list, self.velocity_list, c=colors, s=area, alpha=0.5)
        # plt.plot(step_list, self.velocity_list, linewidth=2, linestyle="-", c="b")
        plt.title('Filtered velocities')
        axes = plt.gca()
        ymin = 0
        ymax = 5000
        axes.set_ylim([ymin, ymax])
        plt.xlabel('frame number')
        plt.ylabel('velocities')
        # axes.set_yscale('log')
        # plt.gca().invert_yaxis()
        plt.savefig("plots/Filtered_velocities_scatter3.png")

    def get_cadence(self):
        """
        Get the cadence from the step indexes
        :return: average cadence
        """
        colors = (0, 0, 0)
        area = 10
        step_list = self.right_foot_index + self.left_foot_index
        step_list.sort()
        save_to_file_list = []
        print(step_list)
        for idx, count in enumerate(step_list):
            try:
                if step_list[idx] == step_list[idx + 1]:
                    del step_list[idx]
            except IndexError:
                pass

        for idx, count in enumerate(step_list):
            if idx == 0:
                pass
            else:
                frames_passed = step_list[idx] - step_list[idx - 1]
                time_passed = frames_passed * (1 / self.fps)
                cadence = 60 / time_passed
                self.cadence.append(cadence)

        mean = np.mean(self.cadence)
        std = np.std(self.cadence)
        save_to_file_list.append("Unfiltered mean: {}".format(mean))
        save_to_file_list.append("Unfiltered std: {}".format(std))
        print("Average cadence: ", mean)
        print("std: ", std)
        print("Cadence", self.cadence)
        print("Step list", step_list)

        plt.figure()
        plt.plot(step_list[1:], self.cadence, linewidth=2, linestyle="-", c="b")
        plt.title('Unfiltered cadence')
        axes = plt.gca()
        ymin = 0
        ymax = 2000
        axes.set_ylim([ymin, ymax])
        plt.xlabel('frame number')
        plt.ylabel('Cadence')
        # axes.set_yscale('log')
        # plt.gca().invert_yaxis()
        plt.savefig("plots/UnfilteredCadence.png")

        plt.figure()
        plt.scatter(step_list[1:], self.cadence, c=colors, s=area, alpha=0.5)
        plt.title('Unfiltered cadence')
        axes = plt.gca()
        ymin = 0
        ymax = 2000
        axes.set_ylim([ymin, ymax])
        plt.xlabel('frame number')
        plt.ylabel('Cadence')
        # axes.set_yscale('log')
        # plt.gca().invert_yaxis()
        plt.savefig("plots/ScatterUnfilteredCadence.png")

        idx_list = []
        data = np.array(self.cadence)
        d = np.abs(data - np.median(data))
        # Get the median and absolute distance to median
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        m = 2
        # print(data[s<2])

        ''' Get the indexes of the outliers'''
        for idx, x in enumerate(self.cadence):
            if s[idx] > m:
                idx_list.append(idx)

        # remove the indexes of outliers
        for index in sorted(idx_list, reverse=True):
            del self.cadence[index]
            del step_list[index]

        mean = np.mean(self.cadence)
        std = np.std(self.cadence)
        save_to_file_list.append("Filtered mean: {}".format(mean))
        save_to_file_list.append("filtered std: {}".format(std))
        print("Average filtered cadence: ", mean)
        print("filtered std: ", std)
        print("Filtered Cadence", self.cadence)

        plt.figure()
        plt.plot(step_list[1:], self.cadence, linewidth=2, linestyle="-", c="b")
        plt.title('Filtered cadence')
        axes = plt.gca()
        ymin = 0
        ymax = 2000
        axes.set_ylim([ymin, ymax])
        plt.xlabel('frame number')
        plt.ylabel('Cadence')
        # axes.set_yscale('log')
        # plt.gca().invert_yaxis()
        plt.savefig("plots/FilteredCadence.png")

        plt.figure()
        plt.scatter(step_list[1:], self.cadence, c=colors, s=area, alpha=0.5)
        plt.title('Filtered cadence')
        axes = plt.gca()
        ymin = 0
        ymax = 2000
        axes.set_ylim([ymin, ymax])
        plt.xlabel('frame number')
        plt.ylabel('Cadence')
        # axes.set_yscale('log')
        # plt.gca().invert_yaxis()
        plt.savefig("plots/ScatterFilteredCadence.png")

        for idx, cadence in enumerate(self.cadence):
            save_to_file_list.append("Frame number: {} Cadence: {}".format(step_list[idx], cadence))
        self.save_text(save_to_file_list, "Cadence")

        # plt.show()


class GUI(QMainWindow):
    """
    Creates the main GUI window for the user
    """

    def __init__(self, display):
        super(GUI, self).__init__()
        self.num_operations = 0
        self.calc = None
        self.output_movie = ""
        self.wid = QWidget(self)
        self.tab = QTabWidget(self)
        self.setCentralWidget(self.tab)
        self.grid = QGridLayout()
        self.grid2 = QGridLayout()
        self.palette = QPalette()
        self.palette.setColor(QPalette.Button, Qt.blue)
        self.palette.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(self.palette)
        self.setGeometry(50, 50, 500, 500)
        self.setWindowTitle("Early development user interface A204 V2.31")
        self.display = display
        self.display.gui = self
        # self.app = QApplication([])
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # self.window = QWidget(parent=self)
        # self.layout = QBoxLayout(QBoxLayout.LeftToRight, self.window)
        """ Create the widgets """
        self.create_tabs()

        self.gif()
        self.gif2()

        self.start_Button()
        self.start_Button2()

        self.dropdown()

        self.print_option_checkbox()
        self.distance_checkbox = Qt.Unchecked
        self.angle_checkbox = Qt.Unchecked
        self.leg_angle_body_checkbox = Qt.Unchecked
        self.left_knee_angle_checkbox = Qt.Unchecked
        self.right_knee_angle_checkbox = Qt.Unchecked

        self.create_step_width_checkbox()
        self.coronal_checkbox = Qt.Unchecked
        self.foot_angle_checkbox = Qt.Unchecked
        self.plot_checkbox()
        self.trajectory_checkbox = Qt.Unchecked

        self.metric_labels()
        self.create_average_step_width_label()

        self.progress_bar()
        self.progress_bar2()

        # self.window.setLayout(self.layout)
        # self.window.show()
        self.tab1.setLayout(self.grid)
        self.tab2.setLayout(self.grid2)
        self.show()

    def create_tabs(self):
        """
        Creates two tabs for sagittal and coronal options
        :return:
        """

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tab.addTab(self.tab1, "Sagittal")
        self.tab.addTab(self.tab2, "Coronal")

    def gif(self):
        """
        Display the walking gif (skeleton currently)
        :return:
        """

        self.movie_label = QLabel()

        self.movie = QMovie("support/skeleton_gif.gif")
        self.movie_label.setMovie(self.movie)

        self.movie.start()
        self.grid.addWidget(self.movie_label, 0, 3)

    def gif2(self):
        """
        Display the walking gif for coronal plane (skeleton currently)
        :return:
        """

        self.movie_label2 = QLabel()

        self.movie2 = QMovie("support/skeleton_walking_coronal.gif")
        self.movie_label2.setMovie(self.movie2)

        self.movie2.start()
        self.grid2.addWidget(self.movie_label2, 0, 3)

    def start_Button(self):
        """
        Create a start button and connect it to start functions
        :return:
        """

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.startbuttonclick)
        self.start_button.clicked.connect(self.start_button_functions)
        self.grid.addWidget(self.start_button, 3, 6)
        # self.start_button.move(300, 400)
        # self.layout.addWidget(self.start_button)

    def start_Button2(self):
        """
        Create a start button for coronal plane tab and connect it to start functions
        :return:
        """
        self.start_button2 = QPushButton('Start', self)
        self.start_button2.clicked.connect(self.startbuttonclick2)
        self.start_button2.clicked.connect(self.start_button_functions2)
        self.grid2.addWidget(self.start_button2, 3, 6)

    @pyqtSlot()
    def start_button_functions2(self):
        """
        Assign worker threads for processing functions in coronal plane
        :return:
        """
        self.worker_thread2 = Worker2(self)
        self.worker_thread2.finish_signal2.connect(self.process_complete_messagebox2)
        self.worker_thread2.start()
        ''' Assign worker signal to function '''
        self.worker_thread2.start_signal2.connect(self.no_option_messagebox)

    @pyqtSlot()
    def start_button_functions(self):
        """
        Assign worker threads for processing functions in the sagittal plane
        :return:
        """
        # Remove any current images in output file
        self.worker_thread = Worker(self)
        self.worker_thread.finish_signal.connect(self.process_complete_messagebox)
        self.worker_thread.start()
        self.worker_thread.start_signal.connect(self.no_option_messagebox)

    @pyqtSlot()
    def no_option_messagebox(self):
        """
        If no options are selected prompt user with a message box
        :return:
        """
        print("No option selected ! ")
        msg = QMessageBox()
        msg.setWindowTitle("Whoops ! ")
        msg.setText("No options were selected ! ")
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_()

    @pyqtSlot()
    def process_complete_messagebox(self):
        """
        Once processing is complete notify the user
        :return:
        """
        print("Process complete ! ")
        msg = QMessageBox.question(self, "The operations have successfully finished ! ",
                                   "Do you want to preview the output video?",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if msg == QMessageBox.Yes:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "processed_video\\Output.avi")
            # print("DIRNAME", path, my_path)
            startfile(path)
        else:
            pass

    @pyqtSlot()
    def process_complete_messagebox2(self):
        """
        Notify the user with a messagebox once processing in the coronal plane is complete
        :return:
        """
        print("Process complete ! ")
        msg = QMessageBox.question(self, "The operations have successfully finished ! ",
                                   "Do you want to preview the output video?",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if msg == QMessageBox.Yes:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "processed_video\\Coronal_Output.avi")
            # print("DIRNAME", path, my_path)
            startfile(path)
        else:
            pass

    pyqtSlot()

    def startbuttonclick2(self):
        """
        Coronal plane assign progress bar thread
        :return:
        """
        if self.coronal_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.foot_angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if not self.calc:
            self.calc = External2(self)
            self.calc.mySignal2.connect(self.onCountChanged2)
            self.calc.start()

        else:
            print("set counter to 0")
            self.calc.progress = 0
            self.calc.count = 0

    def startbuttonclick(self):
        """
        sagittal plane assign progress bar thread
        :return:
        """
        if self.angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.distance_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.leg_angle_body_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.left_knee_angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.right_knee_angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if not self.calc:
            self.calc = External(self)
            self.calc.mySignal.connect(self.onCountChanged)
            self.calc.start()
        else:
            print("set counter to 0")
            self.calc.progress = 0
            self.calc.count = 0

    def dropdown(self):
        """
        Create a drop down to choose which points to calculate distance
        :return:
        """
        self.first_label = QLabel("First point", self)
        self.second_label = QLabel("Second point", self)
        self.second_label.setAlignment(Qt.AlignBottom)
        self.first_label.setAlignment(Qt.AlignBottom)
        self.grid.addWidget(self.first_label, 2, 3)
        self.grid.addWidget(self.second_label, 2, 0)

        # self.first_label.move(50, 220)
        # self.second_label.move(300, 220)
        dropdown1 = QComboBox(self)
        dropdown1.addItem("LBigToe")
        dropdown1.addItem("LWrist")
        dropdown1.addItem("LElbow")
        dropdown1.addItem("LEye")
        dropdown1.addItem("LHeel")
        dropdown1.addItem("LAnkle")
        dropdown1.addItem("LHip")
        dropdown1.addItem("LEar")
        dropdown1.addItem("LShoulder")
        dropdown1.addItem("MidHip")
        dropdown1.addItem("Nose")
        dropdown1.addItem("Neck")

        self.grid.addWidget(dropdown1, 3, 3)
        # dropdown1.move(50, 250)
        # self.layout.addWidget(dropdown1)
        dropdown1.activated[str].connect(self.set_dropdown1)

        dropdown2 = QComboBox(self)
        dropdown2.addItem("RBigToe")
        dropdown2.addItem("RWrist")
        dropdown2.addItem("RElbow")
        dropdown2.addItem("REye")
        dropdown2.addItem("RHeel")
        dropdown2.addItem("RAnkle")
        dropdown2.addItem("RHip")
        dropdown2.addItem("REar")
        dropdown2.addItem("RShoulder")
        dropdown2.addItem("MidHip")
        dropdown2.addItem("Nose")
        dropdown2.addItem("Neck")
        # dropdown2.move(300, 250)
        self.grid.addWidget(dropdown2, 3, 0)
        dropdown2.activated[str].connect(self.set_dropdown2)

    def set_dropdown1(self, text):
        self.display.keypoint1 = text
        print(self.display.keypoint1)

    def set_dropdown2(self, text):
        self.display.keypoint2 = text
        print(self.display.keypoint2)

    def print_option_checkbox(self):
        """
        Check boxes for selecting which measurements to use
        :return:
        """
        self.checkbox_layout = QVBoxLayout()
        label = QLabel("Select your measurements:", self)
        self.checkbox_layout.addWidget(label)

        box = QCheckBox("Distance", self)
        box.stateChanged.connect(self.distance_clickbox)
        self.checkbox_layout.addWidget(box)

        box_angle = QCheckBox("Angle", self)
        box_angle.stateChanged.connect(self.angle_clickbox)
        self.checkbox_layout.addWidget(box_angle)

        leg_body_angle_checkbox = QCheckBox("Leg body angle", self)
        leg_body_angle_checkbox.stateChanged.connect(self.leg_body_angle_clickbox)
        self.checkbox_layout.addWidget(leg_body_angle_checkbox)

        right_knee_angle_checkbox = QCheckBox("Right knee angle", self)
        right_knee_angle_checkbox.stateChanged.connect(self.right_knee_angle_clickbox)
        self.checkbox_layout.addWidget(right_knee_angle_checkbox)

        left_knee_angle_checkbox = QCheckBox("Left knee angle", self)
        left_knee_angle_checkbox.stateChanged.connect(self.left_knee_angle_clickbox)
        self.checkbox_layout.addWidget(left_knee_angle_checkbox)

        temp_widget = QWidget()
        temp_widget.setLayout(self.checkbox_layout)
        self.grid.addWidget(temp_widget, 1, 0)

    def plot_checkbox(self):
        self.trajectory_layout = QVBoxLayout()
        self.trajectory_label = QLabel("Trajectory: ", self)
        self.trajectory_layout.addWidget(self.trajectory_label)

        box = QCheckBox("Plot trajectory", self)
        box.stateChanged.connect(self.trajectory_clickbox)
        self.trajectory_layout.addWidget(box)

        temp_widget = QWidget()
        temp_widget.setLayout(self.trajectory_layout)
        self.grid.addWidget(temp_widget, 1, 4)

    def create_average_step_width_label(self):
        self.coronal_metrics_layout = QVBoxLayout()
        self.average_step_width_label = QLabel("Average step width: ", self)
        self.coronal_metrics_layout.addWidget(self.average_step_width_label)
        temp_widget = QWidget()
        temp_widget.setLayout(self.coronal_metrics_layout)
        self.grid2.addWidget(temp_widget, 1, 0)

    def create_step_width_checkbox(self):
        self.step_width_layout = QVBoxLayout()
        self.step_width_label = QLabel("Select measurements: ", self)
        self.step_width_label.setAlignment(Qt.AlignBottom)
        self.step_width_layout.addWidget(self.step_width_label)

        box = QCheckBox("Get step width", self)
        box.stateChanged.connect(self.coronal_clickbox)
        foot_angle_box = QCheckBox("Get foot angle")
        foot_angle_box.stateChanged.connect(self.foot_angle_clickbox)
        self.step_width_layout.addWidget(box)
        self.step_width_layout.addWidget(foot_angle_box)

        temp_widget = QWidget()
        temp_widget.setLayout(self.step_width_layout)
        self.grid2.addWidget(temp_widget, 2, 4)

    def trajectory_clickbox(self, state):
        if state == Qt.Checked:
            self.trajectory_checkbox = Qt.Checked
            print('Trajectory Checked')
        else:
            self.trajectory_checkbox = Qt.Unchecked
            print('Trajectory Unchecked')

    def angle_clickbox(self, state):
        if state == Qt.Checked:
            self.angle_checkbox = Qt.Checked
            print('Angle Checked')
        else:
            self.angle_checkbox = Qt.Unchecked
            print('Angle Unchecked')

    def leg_body_angle_clickbox(self, state):
        if state == Qt.Checked:
            self.leg_angle_body_checkbox = Qt.Checked
            print('leg body Angle Checked')
        else:
            self.leg_angle_body_checkbox = Qt.Unchecked
            print('leg body Angle Unchecked')

    def right_knee_angle_clickbox(self, state):
        if state == Qt.Checked:
            self.right_knee_angle_checkbox = Qt.Checked
            print('right knee Angle Checked')
        else:
            self.right_knee_angle_checkbox = Qt.Unchecked
            print('Right knee Angle Unchecked')

    def left_knee_angle_clickbox(self, state):
        if state == Qt.Checked:
            self.left_knee_angle_checkbox = Qt.Checked
            print('left knee Angle Checked')
        else:
            self.left_knee_angle_checkbox = Qt.Unchecked
            print('left knee Angle Unchecked')

    def distance_clickbox(self, state):

        if state == Qt.Checked:
            self.distance_checkbox = Qt.Checked
            print('Distance Checked')
        else:
            self.distance_checkbox = Qt.Unchecked
            print('Distance Unchecked')

    def coronal_clickbox(self, state):

        if state == Qt.Checked:
            self.coronal_checkbox = Qt.Checked
            print('Step width Checked')
        else:
            self.coronal_checkbox = Qt.Unchecked
            print('Step width Unchecked')

    def foot_angle_clickbox(self, state):

        if state == Qt.Checked:
            self.foot_angle_checkbox = Qt.Checked
            print('Foot angle Checked')
        else:
            self.foot_angle_checkbox = Qt.Unchecked
            print('Foot angle Unchecked')

    def metric_labels(self):
        self.dist_layout = QVBoxLayout()
        self.max_dist_label = QLabel("Max dist: ", self)
        self.min_dist_label = QLabel("Min dist: ", self)
        self.angle_label = QLabel("Angle: ", self)
        self.dist_layout.addWidget(self.max_dist_label)
        self.dist_layout.addWidget(self.min_dist_label)
        self.dist_layout.addWidget(self.angle_label)
        temp_widget = QWidget()
        temp_widget.setLayout(self.dist_layout)
        self.grid.addWidget(temp_widget, 1, 3)
        # self.grid.addWidget(self.dist_layout, 1, 3)

    def progress_bar(self):
        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 100, 25)
        self.progress.setMaximum(100)
        self.grid.addWidget(self.progress, 4, 0)

    def progress_bar2(self):
        self.progress2 = QProgressBar(self)
        self.progress2.setGeometry(0, 0, 100, 25)
        self.progress2.setMaximum(100)
        self.grid2.addWidget(self.progress2, 4, 0)

    @pyqtSlot(int)
    def onCountChanged(self, value):
        self.progress.setValue(value)

    @pyqtSlot(int)
    def onCountChanged2(self, value):
        self.progress2.setValue(value)


TIME_LIMIT = 2400000000000000000000000000


class External(QThread):
    """
    run the progress bar in an external thread class
    """
    mySignal = pyqtSignal(int)

    def __init__(self, gui):
        super(External, self).__init__()
        self.gui = gui
        # Set number operations to one to avoid divide by 0 error
        if self.gui.num_operations == 0:
            num_operations = 1
        else:
            num_operations = self.gui.num_operations
        # Get number of files to be operated on
        self.num_files = len(self.gui.display.data.data_files) * num_operations
        self.progress = 0
        self.frame = 1
        self.count = 0

    def run(self):
        try:
            # Get progress to add for each number of file
            add = 100 / self.num_files
        except ZeroDivisionError:
            print("ZeroDivisionError")
            sys.exit()
        print("ADD", add)
        # While not timed out
        while self.count < TIME_LIMIT:
            # Add count
            self.count += 1
            # If the frame number is different to processed frame number
            if self.frame != self.gui.display.frame_number:
                print(self.frame, self.gui.display.frame_number, self.progress)
                # Increase the progress bar and set to new frame
                self.frame = self.gui.display.frame_number
                self.progress += add
                # self.gui.onCountChanged(self.progress)
                self.mySignal.emit(self.progress)


# TODO fix coronal plane bugs
class External2(QThread):
    """
    Runs progress bar for coronal plane in external thread
    """
    mySignal2 = pyqtSignal(int)

    def __init__(self, gui, parent=None):
        # super(External2, self).__init__()
        QThread.__init__(self, parent)
        self.gui = gui
        if self.gui.num_operations == 0:
            num_operations = 1
        else:
            num_operations = self.gui.num_operations
        self.num_files = len(self.gui.display.data.coronal_data_files) * num_operations
        self.progress = 0
        self.frame = 1
        self.count = 0

    def run(self):
        try:
            add = 100 / self.num_files
        except ZeroDivisionError:
            print("ZeroDivisionError")
            sys.exit()
        print("ADD", add)
        while self.count < TIME_LIMIT:
            self.count += 1
            if self.frame != self.gui.display.frame_number:
                print(self.frame, self.gui.display.frame_number, self.progress)
                self.frame = self.gui.display.frame_number
                self.progress += add
                # self.gui.onCountChanged2(self.progress)
                print(self.progress)
                self.mySignal2.emit(int(self.progress))


class Worker(QThread):
    """
    Worker thread for processing measurements in the sagittal plane
    """
    start_signal = pyqtSignal()
    finish_signal = pyqtSignal()

    def __init__(self, gui, parent=None):
        # super(External2, self).__init__()
        QThread.__init__(self, parent)
        self.gui = gui

    def run(self):
        files = glob.glob("{}\\*.png".format("output_images"))
        for f in files:
            os.remove(f)
        start = 0
        if self.gui.distance_checkbox == Qt.Checked:
            start = 1
            self.gui.display.distance_overlay()

        if self.gui.angle_checkbox == Qt.Checked:
            start = 1
            self.gui.display.angle_overlay()

        if self.gui.leg_angle_body_checkbox == Qt.Checked:
            start = 1
            self.gui.display.leg_body_angle_overlay()

        if self.gui.left_knee_angle_checkbox == Qt.Checked:
            start = 1
            self.gui.display.left_knee_angle_overlay()

        if self.gui.right_knee_angle_checkbox == Qt.Checked:
            start = 1
            self.gui.display.right_knee_angle_overlay()

        if self.gui.trajectory_checkbox == Qt.Checked:
            start = 1
            self.gui.display.plot_points("RBigToe")

        if start == 0:
            # If no options selected send a signal
            self.start_signal.emit()


        else:
            self.gui.display.display_step_number_overlay()
            for frame in self.gui.display.frame_list:
                save_frame(frame)
            # noinspection PyBroadException
            try:
                save_video()
                self.finish_signal.emit()

            except Exception:
                self.quit()
            # self.process_complete_messagebox()


class Worker2(QThread):
    """
    Worker thread for processing measurements in the coronal plane
    """
    start_signal2 = pyqtSignal()
    finish_signal2 = pyqtSignal()

    def __init__(self, gui, parent=None):
        # super(External2, self).__init__()
        QThread.__init__(self, parent)
        self.gui = gui

    def run(self):
        # Remove any current images in output file
        files = glob.glob("{}\\*.png".format("output_coronal_images"))
        for f in files:
            os.remove(f)
        start = 0

        if self.gui.coronal_checkbox == Qt.Checked:
            start = 1
            self.gui.display.display_step_width()

        if self.gui.foot_angle_checkbox == Qt.Checked:
            start = 1
            self.gui.display.foot_angle_overlay()

        if start == 0:
            self.start_signal2.emit()
        else:
            for frame in self.gui.display.coronal_frame_list:
                save_frame2(frame)
            # noinspection PyBroadException
            try:
                save_video2()
                self.finish_signal2.emit()

            except Exception:
                self.quit()



def save_frame(frame):
    """
    Save a frame to output_images
    :param frame:
    :return:
    """
    time_stamp = datetime.now()
    filename = "{}.png".format(time_stamp.strftime("%Y-%m-%d_%H-%M-%S-%f"))
    path = "output_images\\{}".format(filename)
    cv2.imwrite(path, frame)


def save_frame2(frame):
    """
    Save a frame to output_images
    :param frame:
    :return:
    """
    time_stamp = datetime.now()
    filename = "{}.png".format(time_stamp.strftime("%Y-%m-%d_%H-%M-%S-%f"))
    path = "output_coronal_images\\{}".format(filename)
    cv2.imwrite(path, frame)


def add_line_between_points(frame, points):
    """
    Adds a line overlay between two points and puts pixel distance text
    :param frame:
    :param points:
    :return:
    """
    point1 = list(map(int, points[0]))
    point2 = list(map(int, points[1]))
    cv2.line(frame, tuple(point1), tuple(point2), (0, 255, 0), thickness=3, lineType=8)
    org = tuple(point1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    color = (0, 0, 255)
    thickness = 2
    print("pt1,pt2", point1, point2)
    frame = cv2.putText(frame, 'Distance: {}'.format(get_distance(point1, point2)), org, font,
                        fontscale, color, thickness, cv2.LINE_AA)
    return frame


def add_points_to_image(frame, keypoints):
    """
    Overlay points to an image
    :param frame: Image/frame for points to be overlayed (from extract_frame) in red
    :param keypoints: list of keypoint coordinates to overlay
    :return: writes frame to output_images
    """
    for keypoint in keypoints:
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 10, (0, 0, 255), -1)
    return frame


def get_distance(point1, point2):
    """

    :param point1: list of coordinates
    :param point2:
    :return:
    """
    print("ptt2", point1, point2)
    dist = np.linalg.norm(np.asarray(point1) - np.asarray(point2))
    print("Distance between two points: ", dist)
    return dist


def distance_overlay(display, point1, point2):
    """
    Creates overlay for distance
    :param display:
    :param point1:
    :param point2:
    :return:
    """
    # Remove any current images in output file
    files = glob.glob("{}\\*.png".format("output_images"))
    for f in files:
        os.remove(f)
    # Add overlay
    for idx, path in enumerate(display.data.input_files):
        frame = cv2.imread(path)
        frame = add_points_to_image(frame, [display.fp(point1, idx), display.fp(point2, idx)])
        frame = add_line_between_points(frame, [display.fp(point1, idx), display.fp(point2, idx)])
        save_frame(frame)
    save_video()


def save_video():
    """
    Saves a video of processed image output to processed_video directory
    :return:
    """
    images = []
    for filename in glob.glob("{}\\*.png".format("output_images")):
        images.append(filename)
    try:
        frame = cv2.imread(images[0])
    except IndexError:
        print("Index error: No images in output folder")
        sys.exit("Index error: No images in output folder")
    height, width, layers = frame.shape
    if args['fps']:
        video = cv2.VideoWriter("Output.avi", 0, args['fps'], (width, height))
    else:
        video = cv2.VideoWriter("Output.avi", 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


def save_video2():
    """
    Saves a video of processed image output to coronal processed_video directory
    :return:
    """
    images = []
    for filename in glob.glob("{}\\*.png".format("output_coronal_images")):
        images.append(filename)
    try:
        frame = cv2.imread(images[0])
    except IndexError:
        print("Index error2: No images in output folder")
        sys.exit("Index error: No images in output folder")
    height, width, layers = frame.shape
    # video = cv2.VideoWriter("{}/Coronal_Output.avi".format("processed_video"), 0, 1, (width, height))
    if args['fps']:
        video = cv2.VideoWriter("Coronal_Output.avi", 0, args['fps'], (width, height))
    else:
        video = cv2.VideoWriter("Coronal_Output.avi", 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


def get_mag(pt1):
    return (pt1[0] ** 2 + pt1[1] ** 2) ** 0.5


def get_video_length(video_path):
    """
    Takes a video and return its length (s), frame count and fps
    :param video_path: path to video
    :return: length (seconds), frame count, fps
    """
    print("Video path", video_path)
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0.0:
            print("No video found with video path. Please try again with correct video path")
            sys.exit()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        print('fps = ' + str(fps))
        print('number of frames = ' + str(frame_count))
        print('duration (S) = ' + str(duration))
        minutes = int(duration / 60)
        seconds = duration % 60
        print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
    except ZeroDivisionError:
        print("The video path is wrong.")
        sys.exit()
    return duration, frame_count, fps


def main(argv=None):
    data = ExtractData()
    display = DisplayData(data)
    # distance_overlay(display, "RBigToe", "LBigToe")
    # display.distance_overlay()
    app = QApplication([])
    app.aboutToQuit.connect(app.deleteLater)
    gui = GUI(display)
    app.exec_()


if __name__ == '__main__':
    main(sys.argv[1:])
