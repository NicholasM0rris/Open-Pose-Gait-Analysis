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
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

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


class ExtractData():

    def __init__(self):
        self.cheese = 1
        self.path_to_input = "input_images"
        self.video_dimensions = ""
        self.data_files = []
        self.input_files = []
        self.path = "output"
        self.key_points = defaultdict(list)

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


class DisplayData():

    def __init__(self, data):
        self.data = data
        self.keypoint1 = "LBigToe"
        self.keypoint2 = "RBigToe"

    def fp(self, keypoint, frame_index):
        """
        Returns keypoint as x,y coordinate corresponding to index
        :param keypoint:
        :param frame_index:
        :return:
        """
        return self.data.key_points[keypoint][frame_index][:-1]

    def distance_overlay(self):
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
        for idx, path in enumerate(self.data.input_files):
            frame = cv2.imread(path)
            frame = add_points_to_image(frame, [self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx)])
            frame = add_line_between_points(frame, [self.fp(self.keypoint1, idx), self.fp(self.keypoint2, idx)])
            save_frame(frame)
        save_video()


class GUI(QMainWindow):
    def __init__(self, display):
        super(GUI, self).__init__()
        self.wid = QWidget(self)
        self.setCentralWidget(self.wid)
        self.grid = QGridLayout()
        self.palette = QPalette()
        self.palette.setColor(QPalette.Button, Qt.blue)
        self.palette.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(self.palette)
        self.setGeometry(50, 50, 500, 500)
        self.setWindowTitle("Super cool Alpha ver.")
        self.display = display
        #self.app = QApplication([])
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        #self.window = QWidget(parent=self)
        #self.layout = QBoxLayout(QBoxLayout.LeftToRight, self.window)

        self.gif()

        self.start_Button()

        self.dropdown()

        #self.window.setLayout(self.layout)
        #self.window.show()
        self.wid.setLayout(self.grid)
        self.show()

    def gif(self):

        self.movie_label = QLabel()

        self.movie = QMovie("support/skeleton_gif.gif")
        self.movie_label.setMovie(self.movie)

        self.movie.start()
        self.grid.addWidget(self.movie_label, 0, 3)

    def start_Button(self):

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.display.distance_overlay)
        self.grid.addWidget(self.start_button, 3, 6)
        #self.start_button.move(300, 400)
        #self.layout.addWidget(self.start_button)

    def dropdown(self):
        self.first_label = QLabel("First point", self)
        self.second_label = QLabel("Second point", self)
        self.second_label.setAlignment(Qt.AlignBottom)
        self.first_label.setAlignment(Qt.AlignBottom)
        self.grid.addWidget(self.first_label, 2,3)
        self.grid.addWidget(self.second_label, 2, 0)

        #self.first_label.move(50, 220)
        #self.second_label.move(300, 220)
        dropdown1 = QComboBox(self)
        dropdown1.addItem("LBigToe")
        dropdown1.addItem("LWrist")
        dropdown1.addItem("LElbow")
        self.grid.addWidget(dropdown1, 3, 3)
        #dropdown1.move(50, 250)
        #self.layout.addWidget(dropdown1)
        dropdown1.activated[str].connect(self.set_dropdown1)

        dropdown2 = QComboBox(self)
        dropdown2.addItem("RBigToe")
        dropdown2.addItem("RWrist")
        dropdown2.addItem("RElbow")
        #dropdown2.move(300, 250)
        self.grid.addWidget(dropdown2, 3, 0)
        dropdown2.activated[str].connect(self.set_dropdown2)

    def set_dropdown1(self, text):
        self.display.keypoint1 = text
        print(self.display.keypoint1)

    def set_dropdown2(self, text):
        self.display.keypoint2 = text
        print(self.display.keypoint2)






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
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter("{}/Output.avi".format("processed_video"), 0, 1, (width, height))
    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


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
