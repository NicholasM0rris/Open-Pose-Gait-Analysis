import os
import sys
import glob
import json
from datetime import datetime
import cv2
import numpy as np
from collections import defaultdict
import argparse
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import *
import time
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, help='Add image')
ap.add_argument('-v', '--video', required=False, help='Add Video')
ap.add_argument('-height', '--height', required=False, type=int, help='Add height of the person in centimetres (cm)')
args = vars(ap.parse_args())

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

    def __init__(self):
        self.cheese = 1
        self.path_to_input = "input_images"
        self.path_to_coronal_input = "coronal_input_images"
        self.video_dimensions = ""
        self.data_files = []
        self.input_files = []
        self.coronal_input_files = []
        self.coronal_data_files = []
        self.path = "output"
        self.coronal_path = "coronal_input_data"
        self.key_points = defaultdict(list)
        self.coronal_key_points = defaultdict(list)

        self.get_data_files()
        self.get_data_frames()
        self.extract_frames()

    def print_keypoints(self, key_point=None):
        """
        Prints keypoint list
        :param key_point: Show specific keypoint
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
        print(self.data_files)

    def print_number_data_files(self, df):
        self.cheese = 1
        print(len(df['people'][0]['pose_keypoints_2d']))

    def get_data_frames(self):
        for files in self.data_files:
            temp = []
            temp_df = json.load(open(files))
            for key in key_points.keys():
                self.key_points[key].append(
                    temp_df['people'][0]['pose_keypoints_2d'][key_points[key] * 3:key_points[key] * 3 + 3])
        try:
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
        for filename in glob.glob("{}\\*.png".format(self.path_to_input)):
            self.input_files.append(filename)
        for filename in glob.glob("{}\\*.png".format(self.path_to_coronal_input)):
            self.coronal_input_files.append(filename)


class DisplayData:

    def __init__(self, data):
        self.data = data
        self.keypoint1 = "LBigToe"
        self.keypoint2 = "RBigToe"
        self.gui = None
        self.distances = []
        self.distances.append("Distance from {} to {}".format(self.keypoint1, self.keypoint2))
        self.num_distances = []
        self.angles = []
        self.num_angles = []
        self.step_width = []
        self.num_step_width = []
        self.frame_list = []
        self.coronal_frame_list = []
        self.frame_number = 1

    def plot_points(self, keypoint):
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
        Saves a test file containing the log
        :return:
        """

        time_stamp = datetime.now()
        filename = str("metrics/{}_log_{}.txt".format(text_type, time_stamp.strftime("%Y-%m-%d_%H-%M-%S")))
        f = open(filename, "w+")
        for something in alist:
            f.write("%s\n" % something)
        f.close()
        self.frame_number = 1

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
        Creates overlay for distance
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
        Returns the angle from three points
        :param p1: point 1 (x,y)
        :param p2: point 2 (x,y)
        :param p3: common point to pt 1 & 2 (x,y)
        :return: angle in degrees?
        """
        a = (p1[0] - p2[0], p1[1] - p2[1])
        b = (p1[0] - p3[0], p1[1] - p3[1])
        angle = np.arccos(np.dot(a, b) / (get_mag(a) * get_mag(b)))
        print("ANGLE", angle)
        return np.degrees(angle)

    def angle_overlay(self):
        """
        Creates overlay for distance
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

        if self.gui.angle_checkbox == Qt.Checked:
            print("Saving angles to text file")
            self.save_text(self.angles, "Angle")
        max_angle = 0
        for an_angle in self.num_angles:
            if an_angle > max_angle:
                max_angle = an_angle

        self.gui.angle_label.setText("Max Angle: {}".format(max_angle))

    def get_step_width(self, index):
        """
        Get the coronal plane step width distances
        :return:
        """
        return abs(self.fp2("RHeel", index)[0] - self.fp2("LHeel", index)[0])

    def display_step_width(self):
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
        for dist in self.num_step_width:
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist
        # self.gui.max_dist_label.setText("Max dist: {}".format(max_dist))
        # self.gui.min_dist_label.setText("Min dist: {}".format(min_dist))
        print(min_dist, max_dist)


class GUI(QMainWindow):
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
        self.setWindowTitle("Super cool Alpha ver.")
        self.display = display
        self.display.gui = self
        # self.app = QApplication([])
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # self.window = QWidget(parent=self)
        # self.layout = QBoxLayout(QBoxLayout.LeftToRight, self.window)
        self.create_tabs()

        self.gif()
        self.gif2()

        self.start_Button()
        self.start_Button2()

        self.dropdown()

        self.print_option_checkbox()
        self.distance_checkbox = Qt.Unchecked
        self.angle_checkbox = Qt.Unchecked

        self.step_width_checkbox()
        self.coronal_checkbox = Qt.Unchecked
        self.plot_checkbox()
        self.trajectory_checkbox = Qt.Unchecked

        self.metric_labels()

        self.progress_bar()
        self.progress_bar2()

        # self.window.setLayout(self.layout)
        # self.window.show()
        self.tab1.setLayout(self.grid)
        self.tab2.setLayout(self.grid2)
        self.show()

    def create_tabs(self):

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tab.addTab(self.tab1, "Sagittal")
        self.tab.addTab(self.tab2, "Coronal")

    def gif(self):

        self.movie_label = QLabel()

        self.movie = QMovie("support/skeleton_gif.gif")
        self.movie_label.setMovie(self.movie)

        self.movie.start()
        self.grid.addWidget(self.movie_label, 0, 3)

    def gif2(self):

        self.movie_label2 = QLabel()

        self.movie2 = QMovie("support/skeleton_walking_coronal.gif")
        self.movie_label2.setMovie(self.movie2)

        self.movie2.start()
        self.grid2.addWidget(self.movie_label2, 0, 0)

    def start_Button(self):

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.startbuttonclick)
        self.start_button.clicked.connect(self.start_button_functions)
        self.grid.addWidget(self.start_button, 3, 6)
        # self.start_button.move(300, 400)
        # self.layout.addWidget(self.start_button)

    def start_Button2(self):

        self.start_button2 = QPushButton('Start', self)
        self.start_button2.clicked.connect(self.startbuttonclick2)
        self.start_button2.clicked.connect(self.start_button_functions2)
        self.grid2.addWidget(self.start_button2, 3, 6)

    def start_button_functions2(self):
        # Remove any current images in output file
        files = glob.glob("{}\\*.png".format("output_coronal_images"))
        for f in files:
            os.remove(f)
        start = 0

        if self.coronal_checkbox == Qt.Checked:
            start = 1
            self.display.display_step_width()

        if start == 0:
            print("No option selected ! ")
            msg = QMessageBox()
            msg.setWindowTitle("Whoops ! ")
            msg.setText("No options were selected ! ")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()
        else:
            for frame in self.display.coronal_frame_list:
                save_frame2(frame)
            save_video2()
            print("Process complete ! ")
            msg = QMessageBox()
            msg.setWindowTitle("Operation complete ! ")
            msg.setText("The operations have successfully finished ! ")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()

    def start_button_functions(self):
        # Remove any current images in output file
        files = glob.glob("{}\\*.png".format("output_images"))
        for f in files:
            os.remove(f)
        start = 0
        if self.distance_checkbox == Qt.Checked:
            start = 1
            self.display.distance_overlay()

        if self.angle_checkbox == Qt.Checked:
            start = 1
            self.display.angle_overlay()

        if self.trajectory_checkbox == Qt.Checked:
            start = 1
            self.display.plot_points("RBigToe")

        if start == 0:
            print("No option selected ! ")
            msg = QMessageBox()
            msg.setWindowTitle("Whoops ! ")
            msg.setText("No options were selected ! ")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()
        else:
            for frame in self.display.frame_list:
                save_frame(frame)
            save_video()
            print("Process complete ! ")
            msg = QMessageBox()
            msg.setWindowTitle("Operation complete ! ")
            msg.setText("The operations have successfully finished ! ")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()

    def startbuttonclick2(self):
        if self.coronal_checkbox == Qt.Checked:
            self.num_operations += 1
        if not self.calc:
            self.calc = External2(self)
        else:
            print("set counter to 0")
            self.calc.progress = 0
            self.calc.count = 0

    def startbuttonclick(self):
        if self.angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.distance_checkbox == Qt.Checked:
            self.num_operations += 1
        if not self.calc:
            self.calc = External(self)
        else:
            print("set counter to 0")
            self.calc.progress = 0
            self.calc.count = 0

    def dropdown(self):
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
        self.checkbox_layout = QVBoxLayout()
        box = QCheckBox("Distance", self)
        box.stateChanged.connect(self.distance_clickbox)
        self.checkbox_layout.addWidget(box)

        box_angle = QCheckBox("Angle", self)
        box_angle.stateChanged.connect(self.angle_clickbox)
        self.checkbox_layout.addWidget(box_angle)
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

    def step_width_checkbox(self):
        self.step_width_layout = QVBoxLayout()
        self.step_width_label = QLabel("Step width: ", self)
        self.step_width_label.setAlignment(Qt.AlignBottom)
        self.step_width_layout.addWidget(self.step_width_label)

        box = QCheckBox("Get step width", self)
        box.stateChanged.connect(self.coronal_clickbox)
        self.step_width_layout.addWidget(box)

        temp_widget = QWidget()
        temp_widget.setLayout(self.step_width_layout)
        self.grid2.addWidget(temp_widget, 1, 4)

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
            print('Distance Checked')
        else:
            self.coronal_checkbox = Qt.Unchecked
            print('Distance Unchecked')

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

    def onCountChanged(self, value):
        self.progress.setValue(value)

    def onCountChanged2(self, value):
        self.progress2.setValue(value)


TIME_LIMIT = 2400000000000000000000000000


class External(QThread):
    """
    Runs a counter thread.
    """

    def __init__(self, gui):
        super(External, self).__init__()
        self.gui = gui
        if self.gui.num_operations == 0:
            num_operations = 1
        else:
            num_operations = self.gui.num_operations
        self.num_files = len(self.gui.display.data.data_files) * num_operations
        self.progress = 0
        self.frame = 1
        self.count = 0
        self.start()

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
                self.gui.onCountChanged(self.progress)


class External2(QThread):
    """
    Runs a counter thread.
    """

    def __init__(self, gui):
        super(External2, self).__init__()
        self.gui = gui
        if self.gui.num_operations == 0:
            num_operations = 1
        else:
            num_operations = self.gui.num_operations
        self.num_files = len(self.gui.display.data.coronal_data_files) * num_operations
        self.progress = 0
        self.frame = 1
        self.count = 0
        self.start()

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
                self.gui.onCountChanged2(self.progress)


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
    video = cv2.VideoWriter("{}/Output.avi".format("processed_video"), 0, 1, (width, height))
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
    video = cv2.VideoWriter("{}/Coronal_Output.avi".format("processed_video"), 0, 1, (width, height))
    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


def get_mag(pt1):
    return (pt1[0] ** 2 + pt1[1] ** 2) ** 0.5


def main(argv=None):
    data = ExtractData()
    display = DisplayData(data)
    # distance_overlay(display, "RBigToe", "LBigToe")
    # display.distance_overlay()
    app = QApplication([])
    gui = GUI(display)
    app.exec_()


if __name__ == '__main__':
    main(sys.argv[1:])
