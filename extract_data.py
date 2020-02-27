import sys
import glob
import json
from datetime import datetime
import cv2
import numpy as np
from collections import defaultdict
import argparse

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
        self.path_to_data = ""
        self.video_dimensions = ""
        self.data_files = []
        self.path = "output"
        self.key_points = defaultdict(list)

        self.get_data_files()
        self.get_data_frames()

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

    def extract_frame(self, sec=0):
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




class DisplayData():

    def __init__(self, data):
        self.data = data

    def fp(self, keypoint, frame_index):
        """
        Returns keypoint as x,y coordinate corresponding to index
        :param keypoint:
        :param frame_index:
        :return:
        """
        return self.data.key_points[keypoint][frame_index][:-1]


def save_frame(frame):
    time_stamp = datetime.now()
    filename = "{}.png".format(time_stamp.strftime("%Y-%m-%d_%H-%M-%S"))
    path = "output_images\\{}".format(filename)
    cv2.imwrite(path, frame)


def add_line_between_points(frame, points):
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


def main(argv=None):
    data = ExtractData()
    display = DisplayData(data)
    frame = data.extract_frame()
    frame = add_points_to_image(frame, [display.fp("RBigToe", 0), display.fp("LBigToe", 0)])
    frame = add_line_between_points(frame, [display.fp("RBigToe", 0), display.fp("LBigToe", 0)])
    save_frame(frame)


if __name__ == '__main__':
    main(sys.argv[1:])
