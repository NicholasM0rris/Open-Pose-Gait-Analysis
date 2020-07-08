import sys

import re

from PIL import Image, ImageFilter

import sys
from collections import defaultdict
import glob
import json

from main import args

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

        self.path_to_input = "cam2_output_images"

        self.video_dimensions = ""
        self.data_files = []
        self.input_files = []
        self.coronal_input_files = []
        self.coronal_data_files = []

        self.path = "cam2_output"

        self.key_points = defaultdict(list)
        self.coronal_key_points = defaultdict(list)

        self.get_data_files()
        self.get_data_frames()
        self.extract_frames()

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

    def check_keypoint_visibility(self, key):
        """
        Check Open Pose data points for validality
        :param key: The key point to check (e.g Nose)
        :return: True if enough valid key points, False if too many missing points
        """
        valid_points = 0
        for data_point in self.key_points[key]:
            if data_point[0] != 0 and data_point[1] != 0:
                valid_points += 1
        if valid_points < len(self.key_points[key]) / 1.3:
            print("{} has insufficient valid points".format(key))
            return False
        else:
            return True

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

    def print_data_files(self):
        """
        Print current list of data files
        """
        print(self.data_files)

    def print_number_data_files(self, df):
        """
        Print number of data files
        :param df: Data frame / list
        """
        self.cheese = 1
        print(len(df['people'][0]['pose_keypoints_2d']))

    def get_data_frames(self):
        """
        Extract the list of joint keypoint locations from JSON files stored in the list of paths in data_files
        """
        try:
            # Get sagittal plane
            for files in self.data_files:
                temp = []
                temp_df = json.load(open(files))
                for key in key_points.keys():
                    # Check case where no person is present
                    if temp_df['people']:
                        # print("key", key)
                        # print("temp", temp_df)
                        # print(temp_df['people'][0])
                        # print(temp_df['people'][0]['pose_keypoints_2d'])
                        # print(temp_df['people'][0]['pose_keypoints_2d'][key_points[key]])
                        self.key_points[key].append(
                            temp_df['people'][0]['pose_keypoints_2d'][key_points[key] * 3:key_points[key] * 3 + 3])
                    # No person case handling
                    else:
                        self.key_points[key].append([0, 0, 0])

        # Except index error due to empty directory
        except Exception as e:
            raise e
            # TODO investigate this

    def extract_frames(self):
        """
        Extract the open pose image files for sagittal and coronal planes
        """
        for filename in glob.glob("{}\\*.png".format(self.path_to_input)):
            self.input_files.append(filename)


def anonymise_images(frames, nose_points, right_ear_points, left_ear_points):
    """
    Add a Gaussian blur to conceal face for privacy reasons
    :param frames: The list of frames to process (Sagittal plane)
    :param nose: The key point to center the blur
    :return: List of path names for the blurred images
    """
    # print(nose_points)
    # print('nose',len(nose_points))
    # print('frames', len(frames))

    padx = 100
    pady = 80
    for idx, path in enumerate(frames):
        # print(len(frames))
        # print(frames)

        frame = Image.open(path)
        # print(nose_points[idx])
        if nose_points[idx][0] != 0 and nose_points[idx][1] != 0:
            nose_x = nose_points[idx][0]
            nose_y = nose_points[idx][1]
            # print(nose_x, nose_y)

        elif left_ear_points[idx][0] != 0 and left_ear_points[idx][1] != 0:
            nose_x = left_ear_points[idx][0]
            nose_y = left_ear_points[idx][1]
        elif right_ear_points[idx][0] != 0 and right_ear_points[idx][1] != 0:
            nose_x = right_ear_points[idx][0]
            nose_y = right_ear_points[idx][1]
        else:
            # No point available
            nose_x = 0
            nose_y = 0

        point1 = nose_x - padx
        point2 = nose_y - pady
        if point1 < 0:
            point1 = 0
        if point2 < 0:
            point2 = 0

        nose = (int(point1), int(point2), int(nose_x + padx), int(nose_y + pady))
        cropped_frame = frame.crop(nose)
        blurred_frame = cropped_frame.filter(ImageFilter.GaussianBlur(radius=20))
        # print(nose)
        # sys.exit()
        frame.paste(blurred_frame, nose)

        outpath = "{}\\{}.png".format("blurred_images", idx + 1)
        print(outpath)
        print(idx, "idx")
        print("path", path)
        frame.save(outpath)
        if idx == 350:
            print(frames)
            print(len(frames))

    input_files = []
    print("Finished FOR loop")
    for filename in glob.glob("{}\\*.png".format("blurred_images")):
        input_files.append(filename)
    # print(input_files)
    # Stupid python input_files.sort(key=lambda x: int(float(os.path.basename(x).split('.')[0][1:])))
    print("Start sorting")
    input_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print("Finished blurring")
    return input_files


def main(argv=None):
    data = ExtractData()
    print(len(data.data_files))
    print(len(data.key_points["RBigToe"]))
    print(data.key_points["REar"])
    for key in data.key_points:
        print(key, len(data.key_points[key]))
    input_files = anonymise_images(data.input_files, [item[:-1] for item in data.key_points["Nose"]],
                                   [item[:-1] for item in data.key_points["REar"]],
                                   [item[:-1] for item in data.key_points["LEar"]])


if __name__ == '__main__':
    main(sys.argv[1:])
