import cv2
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')
import support.base_functions as bf
import numpy as np
from datetime import datetime
from PyQt5.QtCore import *
from main import args

import sys


class DisplayData:
    """
    The class calculates and displays the processed data and measurements
    """

    def __init__(self, data):
        self.detected_frame_list = []
        self.right_foot_count = 0
        self.left_foot_count = 0
        # Define list for index/frames in which step is made
        self.right_foot_index = []
        self.left_foot_index = []
        self.data = data
        # print(self.data.input_files)
        # Blur face for anonymity
        if args["anon"] == 'True':
            if self.data.check_keypoint_visibility("REar"):
                print("ear", len(self.data.input_files))
                self.data.input_files = bf.anonymise_images(self.data.input_files,
                                                            [item[:-1] for item in self.data.key_points["REar"]])
            elif self.data.check_keypoint_visibility("LEar"):
                self.data.input_files = bf.anonymise_images(self.data.input_files,
                                                            [item[:-1] for item in self.data.key_points["LEar"]])
            elif self.data.check_keypoint_visibility("Nose"):
                self.data.input_files = bf.anonymise_images(self.data.input_files,
                                                            [item[:-1] for item in self.data.key_points["Nose"]])
            else:
                print("BAD DATA! Failure attempting to blur faces!")

            # print("data", self.data.input_files)
        if args["anonc"] == 'True':
            self.data.input_files = bf.anonymise_coronal_images(self.data.input_files,
                                                                [item[:-1] for item in self.data.key_points["Nose"]])

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
        # Define angles made at heel and toe
        self.right_ankle_angles = []
        self.num_right_ankle_angles = []
        self.left_ankle_angles = []
        self.num_left_ankle_angles = []
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
        self.duration, self.frame_count, self.fps = bf.get_video_length(self.video_path)
        # Get number of steps
        self.velocity_list = []
        self.stride_length_list = []
        self.cadence = []
        self.left_index_list = []
        self.right_index_list = []
        self.correct_leg_swap()
        if args['treadmill'] == 'True':
            self.get_number_steps()
            self.display_step_number_overlay()
        else:
            self.horizontal_foot_angle_overlay(True)
            self.display_normal_step_number_overlay()
        # self.horizontal_foot_angle_overlay(True)
        # sys.exit()
        # In testing stage
        try:
            self.get_velocity()
            self.get_cadence()
        except:
            pass

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

    # fp stands for fetch point to quickly retreive a keypoint coordinate
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

    def add_line_between_points(self, frame, points, thickness, colour=None):
        """
        Adds a line overlay between two points and puts pixel distance text
        :param colour: The colour of the line
        :param thickness: Thickness of the line
        :param frame: What frame to add the line over
        :param points: Two points to add the line between
        """
        if colour == None:
            colour = (0, 255, 0)

        point1 = list(map(int, points[0]))
        point2 = list(map(int, points[1]))
        cv2.line(frame, tuple(point1), tuple(point2), colour, thickness=thickness, lineType=8)

        # print("pt1,pt2", point1, point2)

        return frame

    def distance_overlay(self):
        """
        Creates overlay for distance over the image frames
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
                dist = bf.get_distance(self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx))
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
                dist = bf.get_distance(self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx))
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

    def get_angle(self, p3, p2, p1, print_option=None):
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
        denominator = (bf.get_mag(a) * bf.get_mag(b))
        angle = np.arccos(numerator / denominator)
        if print_option:
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
                                                     [self.fp("LKnee", idx), self.fp("MidHip", idx)], 16, (255, 0, 0))
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
        right_knee_direction = 1
        left_knee_direction = 1
        ''' Define initial x1 and x2 '''
        RKnee_x1 = self.fp("RKnee", 0)[0]
        RKnee_x2 = self.fp("RKnee", 1)[0]
        RKnee_x3 = self.fp("RKnee", 2)[0]
        if RKnee_x1 > RKnee_x2:
            right_knee_direction = 0
        elif RKnee_x1 < RKnee_x2:
            right_knee_direction = 1
        ''' Get the rate of change '''
        change1 = RKnee_x2 - RKnee_x1
        change2 = RKnee_x3 - RKnee_x2
        rate_change = [change1 - change2]
        average_rate_change = 0
        directions = dict()

        ''' Iterate over the data files to look for leg swaps '''
        for idx, path in enumerate(self.data.input_files):
            try:

                RKnee_x1 = self.fp("RKnee", idx)[0]
                RKnee_x2 = self.fp("RKnee", idx + 1)[0]
                RKnee_x3 = self.fp("RKnee", idx + 2)[0]
                if RKnee_x1 > RKnee_x2:
                    right_knee_direction = 0
                elif RKnee_x1 < RKnee_x2:
                    right_knee_direction = 1
                directions[idx] = right_knee_direction
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
        for idx, frame in enumerate(self.data.input_files):
            try:
                ''' If a high rate change is detected swap the legs'''
                # print('directions',directions)
                if (rate_change[idx] > average_rate_change * 3):  # and (directions[idx-1] != directions[idx]):
                    self.detected_frame_list.append(idx)
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
                    self.data.key_points["LKnee"][idx][:-1], self.data.key_points["RKnee"][idx][:-1] = RKnee, LKnee
                    # LKnee, RKnee = RKnee, LKnee
                    self.data.key_points["LAnkle"][idx][:-1], self.data.key_points["RAnkle"][idx][:-1] = RAnkle, LAnkle
                    # LAnkle, RAnkle = RAnkle, LAnkle
                    self.data.key_points["LBigToe"][idx][:-1], self.data.key_points["RBigToe"][idx][
                                                               :-1] = RBigToe, LBigToe
                    # LBigToe, RBigToe = RBigToe, LBigToe
                    self.data.key_points["LSmallToe"][idx][:-1], self.data.key_points["RSmallToe"][idx][
                                                                 :-1] = RSmallToe, LSmallToe
                    # LSmallToe, RSmallToe = RSmallToe, LSmallToe
                    self.data.key_points["LHeel"][idx][:-1], self.data.key_points["RHeel"][idx][:-1] = RHeel, LHeel
                    # LHeel, RHeel = RHeel, LHeel

            except IndexError:
                pass
        self.save_text(self.detected_frame_list, "Detected_leg_swap_frame_list")

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
                                if idx in self.detected_frame_list:
                                    break
                                y1_r = self.fp("RHeel", idx + i)[1]
                                y2_r = self.fp("RHeel", idx + 1 + i)[1]
                                rate = abs(y1_r - y2_r)
                                i += 1
                                if rate < initial_rate / 1.8:
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
                                # TODO This causes steps to be missed
                                if idx in self.detected_frame_list:
                                    break
                                y1_left = self.fp("LHeel", idx + i)[1]
                                y2_left = self.fp("LHeel", idx + 1 + i)[1]
                                rate = abs(y1_left - y2_left)
                                i += 1
                                if rate < initial_rate / 1.8:
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

    def horizontal_foot_angle_overlay(self, display):
        """
        Creates overlay for angle wrt to horizontal line at heel
        :param display: If display is added, create an overlay on the video, otherwise just find the angle
        :return:
        """
        # Add overlay
        if not self.frame_list:
            # initial right toe and heel coordinates
            right_toe_position = self.fp("RBigToe", 0)
            right_heel_position = self.fp("RHeel", 0)
            # initial left toe and heel coordinates
            left_toe_position = self.fp("LBigToe", 0)
            left_heel_position = self.fp("LHeel", 0)
            index_list = set()
            left_index_list = set()
            initial_phase = 0
            left_phase = 0

            # Initial phase = 1 when toe is above heel, 0 when below
            if right_toe_position < right_heel_position:
                initial_phase = 1
            else:
                initial_phase = 0

            if left_toe_position < left_heel_position:
                left_phase = 1
            else:
                left_phase = 0

            for idx, path in enumerate(self.data.input_files):

                angle = self.get_angle(self.fp("RBigToe", idx), self.fp("RHeel", idx),
                                       (self.fp("RHeel", idx)[0] + 200, self.fp("RHeel", idx)[1]))
                left_angle = self.get_angle(self.fp("LBigToe", idx), self.fp("LHeel", idx),
                                            (self.fp("LHeel", idx)[0] + 200, self.fp("LHeel", idx)[1]))

                self.right_ankle_angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_right_ankle_angles.append(angle)

                self.left_ankle_angles.append("Frame {} - Angle: {}".format(self.frame_number, left_angle))
                self.num_left_ankle_angles.append(left_angle)

                right_toe_position = self.fp("RBigToe", idx)
                right_heel_position = self.fp("RHeel", idx)
                if right_toe_position[1] < right_heel_position[1] and angle > 25:
                    initial_phase = 1
                else:
                    initial_phase = 0
                left_toe_position = self.fp("LBigToe", idx)
                left_heel_position = self.fp("LHeel", idx)
                if left_toe_position[1] < left_heel_position[1] and left_angle > 25:
                    left_phase = 1
                else:
                    left_phase = 0

                # Check if the toe is above the heel, and the angle is greater than 25 degrees
                # print("Angle {}\n Phase {}".format(angle, initial_phase))
                # if right_toe_position[1] < right_heel_position[1]:

                #   print("Toe above at idx {}".format(idx))
                if right_toe_position[1] < right_heel_position[1] and angle > 25:
                    # The trajectory of incoming step placement detected -
                    error_count = 0
                    for i in range(1, 6):
                        # Check if the toe is above  in next five frames
                        try:
                            new_right_toe_position = self.fp("RBigToe", idx + i)
                            new_right_heel_position = self.fp("RHeel", idx + i)

                            if new_right_heel_position[1] < new_right_toe_position[1]:
                                error_count += 1
                            if new_right_toe_position[0] - self.fp("RBigToe", idx + i - 1)[0] > 5 * (
                                    self.fp("RBigToe", idx + i - 1)[0] - self.fp("RBigToe", idx + i - 2)[0]):
                                error_count += 1
                            if error_count >= 2:
                                # print("Error break at idx {}".format(idx + i))
                                initial_phase = 0
                                break
                            angle = self.get_angle(self.fp("RBigToe", idx + i), self.fp("RHeel", idx + i),
                                                   (self.fp("RHeel", idx + i)[0] + 200, self.fp("RHeel", idx + i)[1]))
                            print("New angle", angle)
                            if angle <= 20 and new_right_heel_position[1] > new_right_toe_position[1] and (
                                    new_right_toe_position[0] - self.fp("RBigToe", idx + i - 1)[0]) < (
                                    5 * (self.fp("RBigToe", idx + i - 1)[0] - self.fp("RBigToe", idx + i - 2)[0])):
                                # print("heel {}, toe {}".format(new_right_heel_position, new_right_toe_position))
                                # print(idx + i)
                                # print("\nPOSSIBLE STEP AT", idx + i, "\n")
                                index_list.add(idx + i)
                                initial_phase = 0
                                break
                        except IndexError:
                            pass
                # print("right indexlist", index_list)

                if left_toe_position[1] < left_heel_position[1] and angle > 25:
                    # The trajectory of incoming step placement detected -
                    error_count = 0
                    for i in range(1, 6):
                        # Check if the toe is above  in next five frames
                        try:
                            new_left_toe_position = self.fp("LBigToe", idx + i)
                            new_left_heel_position = self.fp("LHeel", idx + i)

                            if new_left_heel_position[1] < new_left_toe_position[1]:
                                error_count += 1
                            if new_left_toe_position[0] - self.fp("LBigToe", idx + i - 1)[0] > 5 * (
                                    self.fp("LBigToe", idx + i - 1)[0] - self.fp("LBigToe", idx + i - 2)[0]):
                                error_count += 1
                            if error_count >= 2:
                                # print("Error break at idx {}".format(idx + i))
                                left_phase = 0
                                break
                            angle = self.get_angle(self.fp("LBigToe", idx + i), self.fp("LHeel", idx + i),
                                                   (self.fp("LHeel", idx + i)[0] + 200, self.fp("LHeel", idx + i)[1]))
                            print("New angle", angle)
                            if angle <= 20 and new_left_heel_position[1] > new_left_toe_position[1] and (
                                    new_left_toe_position[0] - self.fp("LBigToe", idx + i - 1)[0]) < (
                                    5 * (self.fp("LBigToe", idx + i - 1)[0] - self.fp("LBigToe", idx + i - 2)[0])):
                                # print("heel {}, toe {}".format(new_right_heel_position, new_right_toe_position))
                                # print(idx + i)
                                # print("\nPOSSIBLE STEP AT", idx + i, "\n")
                                left_index_list.add(idx + i)
                                left_phase = 0
                                break
                        except IndexError:
                            pass
                # print("left indexlist", left_index_list)

                if display:
                    frame = cv2.imread(path)
                    frame = self.add_points_to_image(frame,
                                                     [self.fp("RHeel", idx), self.fp("RBigToe", idx),
                                                      [self.fp("RHeel", idx)[0] + 200, self.fp("RHeel", idx)[1]]])
                    frame = self.add_line_between_points(frame,
                                                         [self.fp("RHeel", idx),
                                                          [self.fp("RHeel", idx)[0] + 200, self.fp("RHeel", idx)[1]]],
                                                         16)
                    frame = self.add_line_between_points(frame,
                                                         [self.fp("RHeel", idx), self.fp("RBigToe", idx)], 16)
                    org = tuple([int(self.fp("RBigToe", idx)[0] + 200), int(self.fp("RBigToe", idx)[1])])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontscale = 1
                    color = (0, 0, 255)
                    thickness = 2
                    frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                        fontscale, color, thickness, cv2.LINE_AA)
                    self.frame_number += 1
                    # save_frame(frame)
                    self.frame_list.append(frame)

        else:
            """ If frame list is not empty, use the existing frames """
            right_toe_position = self.fp("RBigToe", 0)
            right_heel_position = self.fp("RHeel", 0)
            # initial left toe and heel coordinates
            left_toe_position = self.fp("LBigToe", 0)
            left_heel_position = self.fp("LHeel", 0)
            index_list = []
            left_index_list = []
            temp_list = []
            initial_phase = 0
            left_phase = 0

            # Initial phase = 1 when toe is above heel, 0 when below
            if right_toe_position < right_heel_position:
                initial_phase = 1
            else:
                initial_phase = 0

            if left_toe_position < left_heel_position:
                left_phase = 1
            else:
                left_phase = 0

            for idx, path in enumerate(self.frame_list):

                angle = self.get_angle(self.fp("RBigToe", idx), self.fp("RHeel", idx),
                                       (self.fp("RHeel", idx)[0] + 200, self.fp("RHeel", idx)[1]))
                left_angle = self.get_angle(self.fp("LBigToe", idx), self.fp("LHeel", idx),
                                            (self.fp("LHeel", idx)[0] + 200, self.fp("LHeel", idx)[1]))

                self.right_ankle_angles.append("Frame {} - Angle: {}".format(self.frame_number, angle))
                self.num_right_ankle_angles.append(angle)

                self.left_ankle_angles.append("Frame {} - Angle: {}".format(self.frame_number, left_angle))
                self.num_left_ankle_angles.append(left_angle)

                right_toe_position = self.fp("RBigToe", idx)
                right_heel_position = self.fp("RHeel", idx)
                if right_toe_position[1] < right_heel_position[1] and angle > 25:
                    initial_phase = 1
                else:
                    initial_phase = 0
                left_toe_position = self.fp("LBigToe", idx)
                left_heel_position = self.fp("LHeel", idx)
                if left_toe_position[1] < left_heel_position[1] and left_angle > 25:
                    left_phase = 1
                else:
                    left_phase = 0

                # Check if the toe is above the heel, and the angle is greater than 25 degrees
                # print("Angle {}\n Phase {}".format(angle, initial_phase))
                # if right_toe_position[1] < right_heel_position[1]:

                #   print("Toe above at idx {}".format(idx))
                if right_toe_position[1] < right_heel_position[1] and angle > 25:
                    # The trajectory of incoming step placement detected -
                    error_count = 0
                    for i in range(1, 6):
                        # Check if the toe is above  in next five frames
                        try:
                            new_right_toe_position = self.fp("RBigToe", idx + i)
                            new_right_heel_position = self.fp("RHeel", idx + i)

                            if new_right_heel_position[1] < new_right_toe_position[1]:
                                error_count += 1
                            if new_right_toe_position[0] - self.fp("RBigToe", idx + i - 1)[0] > 5 * (
                                    self.fp("RBigToe", idx + i - 1)[0] - self.fp("RBigToe", idx + i - 2)[0]):
                                error_count += 1
                            if error_count >= 2:
                                # print("Error break at idx {}".format(idx + i))
                                initial_phase = 0
                                break
                            angle = self.get_angle(self.fp("RBigToe", idx + i), self.fp("RHeel", idx + i),
                                                   (self.fp("RHeel", idx + i)[0] + 200, self.fp("RHeel", idx + i)[1]))
                            print("New angle", angle)
                            if angle <= 20 and new_right_heel_position[1] > new_right_toe_position[1] and (
                                    new_right_toe_position[0] - self.fp("RBigToe", idx + i - 1)[0]) < (
                                    5 * (self.fp("RBigToe", idx + i - 1)[0] - self.fp("RBigToe", idx + i - 2)[0])):
                                # print("heel {}, toe {}".format(new_right_heel_position, new_right_toe_position))
                                # print(idx + i)
                                # print("\nPOSSIBLE STEP AT", idx + i, "\n")
                                index_list.add(idx + i)
                                initial_phase = 0
                                break
                        except IndexError:
                            pass
                print("right indexlist", index_list)

                if left_toe_position[1] < left_heel_position[1] and angle > 25:
                    # The trajectory of incoming step placement detected -
                    error_count = 0
                    for i in range(1, 6):
                        # Check if the toe is above  in next five frames
                        try:
                            new_left_toe_position = self.fp("LBigToe", idx + i)
                            new_left_heel_position = self.fp("LHeel", idx + i)

                            if new_left_heel_position[1] < new_left_toe_position[1]:
                                error_count += 1
                            if new_left_toe_position[0] - self.fp("LBigToe", idx + i - 1)[0] > 5 * (
                                    self.fp("LBigToe", idx + i - 1)[0] - self.fp("LBigToe", idx + i - 2)[0]):
                                error_count += 1
                            if error_count >= 2:
                                # print("Error break at idx {}".format(idx + i))
                                left_phase = 0
                                break
                            angle = self.get_angle(self.fp("LBigToe", idx + i), self.fp("LHeel", idx + i),
                                                   (self.fp("LHeel", idx + i)[0] + 200, self.fp("LHeel", idx + i)[1]))
                            print("New angle", angle)
                            if angle <= 20 and new_left_heel_position[1] > new_left_toe_position[1] and (
                                    new_left_toe_position[0] - self.fp("LBigToe", idx + i - 1)[0]) < (
                                    5 * (self.fp("LBigToe", idx + i - 1)[0] - self.fp("LBigToe", idx + i - 2)[0])):
                                # print("heel {}, toe {}".format(new_right_heel_position, new_right_toe_position))
                                # print(idx + i)
                                # print("\nPOSSIBLE STEP AT", idx + i, "\n")
                                left_index_list.add(idx + i)
                                left_phase = 0
                                break
                        except IndexError:
                            pass
                print("left indexlist", left_index_list)


                if display:
                    frame = self.add_points_to_image(frame,
                                                     [self.fp("RHeel", idx), self.fp("RBigToe", idx),
                                                      (self.fp("RHeel", idx)[0] + 200, self.fp("RHeel", idx)[1])])
                    frame = self.add_line_between_points(frame,
                                                         [self.fp("RHeel", idx), self.fp("RHeel", idx)[0] + 200,
                                                          self.fp("RHeel", idx)[1]], 16)
                    frame = self.add_line_between_points(frame,
                                                         [self.fp("RHeel", idx), self.fp("RBigToe", idx)], 16)
                    org = tuple([int(self.fp("RBigToe", idx)[0] + 200), int(self.fp("RBigToe", idx)[1])])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontscale = 1
                    color = (0, 0, 255)
                    thickness = 2

                    frame = cv2.putText(frame, 'Angle: {}'.format(angle), org, font,
                                        fontscale, color, thickness, cv2.LINE_AA)
                    self.frame_number += 1
                    temp_list.append(frame)
            if display:
                self.frame_list = temp_list

        left_index_list = sorted(list(left_index_list))
        del_list = []
        for indx, i in enumerate(left_index_list):
            try:
                if i + 30 > left_index_list[indx + 1]:
                    count = 1
                    while True:
                        if i + 30 > left_index_list[indx + count]:
                            print("Del {}".format(left_index_list[indx + count]))
                            del_list.append(left_index_list[indx + count])
                            count += 1
                        else:
                            break

            except IndexError:
                print("LIdx reached")
                pass
        print(del_list)
        for num in del_list:
            try:
                left_index_list.remove(num)
            except ValueError:
                pass

        index_list = sorted(list(index_list))
        del_list = []
        for indx, i in enumerate(index_list):
            try:
                if i + 30 > index_list[indx + 1]:
                    count = 1
                    while True:
                        if i + 30 > index_list[indx + count]:
                            print("Del {}".format(index_list[indx + count]))
                            del_list.append(index_list[indx + count])
                            count += 1
                        else:
                            break

            except IndexError:
                print("RIdx reached")
                pass
        for num in del_list:
            try:
                index_list.remove(num)
            except ValueError:
                pass
        print("Final left index", left_index_list)
        print("Final right index", index_list)

        self.left_index_list = left_index_list
        self.right_index_list = index_list

        self.save_text(self.right_ankle_angles, "Right_Horizontal_foot_angle")
        self.save_text(self.left_ankle_angles, "Left_Horizontal_Foot_Angle")

    def display_normal_step_number_overlay(self):

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
                if idx in self.left_index_list:
                    left_foot_count += 1
                if idx in self.right_index_list:
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
                if idx in self.left_index_list:
                    left_foot_count += 1
                if idx in self.right_index_list:
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
        if args['treadmill'] == 'True':
            step_list = self.right_foot_index + self.left_foot_index
            step_list.sort()
        else:
            step_list = self.right_index_list + self.left_index_list
            step_list.sort()

        for x, idx in enumerate(step_list):
            try:
                displacement = abs(bf.get_distance(self.fp("LBigToe", idx), self.fp("RBigToe", idx)))
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
        if args['treadmill'] == 'True':
            step_list = self.right_foot_index + self.left_foot_index
            step_list.sort()
        else:
            step_list = self.right_index_list + self.left_index_list
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
