import os
import sys
from collections import defaultdict
import glob
import json
import cv2
from main import args

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
                            if (tuple(self.key_points[key][idx + 1]) != (0, 0, 0)) and (
                                    tuple(self.key_points[key][idx + 2]) != (0, 0, 0)):
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
                                    for indx, i in enumerate(reversed(range(iterator + 1))):
                                        self.key_points[key][i][0] -= diff_x * (indx + 1)
                                        self.key_points[key][i][1] -= diff_y * (indx + 1)
                                except IndexError:
                                    pass
                        # Last frame - need to extrapolate
                        elif idx == len(self.key_points[key]) - 1:
                            item[0] = 2 * self.key_points[key][idx - 1][0] - self.key_points[key][idx - 2][0]
                            item[1] = 2 * self.key_points[key][idx - 1][1] - self.key_points[key][idx - 2][1]
                        # Else interpolate average of f-1 and f+1
                        else:
                            # print("Changing index {} item {} to ".format(idx, item))
                            # If the frame after the missing data point is valid: interpolate using that point
                            if tuple(self.key_points[key][idx + 1]) != (0, 0, 0):
                                item[0] = (self.key_points[key][idx - 1][0] + self.key_points[key][idx + 1][0]) / 2
                                item[1] = (self.key_points[key][idx - 1][1] + self.key_points[key][idx + 1][1]) / 2
                            # print("this", item)
                            # However if the point afterwords is also missing, search for non missing point
                            else:
                                print("True", idx, key)
                                frame_x = self.key_points[key][idx + 1][0]
                                frame_y = self.key_points[key][idx + 1][1]
                                iterator = 0
                                try:
                                    while frame_x == 0 and frame_y == 0:
                                        iterator += 1
                                        # Iterate over frames until the points are no longer missing
                                        frame_x = self.key_points[key][idx + 1 + iterator][0]
                                        frame_y = self.key_points[key][idx + 1 + iterator][1]

                                    diff_x = (frame_x - self.key_points[key][idx-1][0]) / (iterator + 2)
                                    diff_y = (frame_y - self.key_points[key][idx-1][1]) / (iterator + 2)

                                    for indx, inter in enumerate(reversed(range(iterator+1))):

                                        self.key_points[key][idx + inter][0] = frame_x - diff_x * (indx + 1)
                                        self.key_points[key][idx + inter][1] = frame_y - diff_y * (indx + 1)
                                    ''' FOR TESTING
                                    if key == "LKnee":
                                        print('framex, framey, iterator, diffx, diffy', frame_x, frame_y, iterator,
                                              diff_x, diff_y)
                                        print(self.key_points[key][idx][0], self.key_points[key][idx][1])
                                        print('indx, inter', indx, inter)
                                        print('idx: ', idx)
                                        print('pre', self.key_points[key][idx - 1][0],
                                              self.key_points[key][idx - 1][1])
                                        print('curret', self.key_points[key][idx][0],
                                              self.key_points[key][idx][1])
                                        print('after1', self.key_points[key][idx + 1][0],
                                              self.key_points[key][idx + 1][1])
                                        print('after2', self.key_points[key][idx + 2][0],
                                              self.key_points[key][idx + 2][1])
                                        sys.exit()
                                    '''
                                except IndexError:
                                    print("Reach end of index while interpolating")
                                    pass


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
