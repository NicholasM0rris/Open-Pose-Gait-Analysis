import os
import sys
import platform
import cv2
import numpy as np
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import glob
import argparse

# Import Openpose (Windows)
dir_path = os.path.dirname(os.path.realpath(__file__))
# Need Open Pose installed to use
'''
try:
    # Windows Import
    if platform.system() == "Windows":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/python/openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except Exception as e:
    raise e
'''
# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/",
                    help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

params = dict()
params["model_folder"] = "../../../models/"
params["number_people_max"] = 1

# Who knows what this does, since if we look at the open pose documentation we see that... It sucks.
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
        if key not in params: params[key] = next_item


class GUI(QMainWindow):
    """
    Creates the main GUI window for the user
    """

    def __init__(self):
        super(GUI, self).__init__()
        self.wid = QWidget(self)
        self.grid = QGridLayout()
        self.param_grid = QGridLayout()
        # self.wid.setLayout(self.grid)
        # self.setCentralWidget(self.wid)
        self.tab = QTabWidget(self)
        self.setCentralWidget(self.tab)
        self.setWindowTitle("Early development user interface A204 V2.31")
        self.duration = 'n/a'
        self.fps = 'n/a'
        self.frame_count = 'n/a'
        self.setGeometry(50, 50, 400, 150)

        self.make_tabs()
        self.make_param_labels()
        self.input()
        self.output()
        self.make_start_button()

        self.tab1.setLayout(self.grid)
        self.tab2.setLayout(self.param_grid)
        self.show()

    def make_tabs(self):
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tab.addTab(self.tab1, "OP video")
        self.tab.addTab(self.tab2, "Params")

    def input(self):
        self.input_label = QLabel()
        self.input_label.setText("Input directory: ")
        self.grid.addWidget(self.input_label, 0, 0)

        self.input_lineedit = QLineEdit(self)
        self.grid.addWidget(self.input_lineedit, 0, 1)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.open_input)
        self.grid.addWidget(self.browse_button, 0, 2)

    def output(self):
        self.output_label = QLabel()
        self.output_label.setText("Output location: ")
        self.grid.addWidget(self.output_label, 1, 0)

        self.output_lineedit = QLineEdit(self)
        self.grid.addWidget(self.output_lineedit, 1, 1)

        self.output_browse_button = QPushButton("Browse")
        self.output_browse_button.clicked.connect(self.open_output)
        self.grid.addWidget(self.output_browse_button, 1, 2)

    def open_input(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open directory')
        self.input_path = fname
        self.input_lineedit.setText(self.input_path)

    def open_output(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open output directory')
        self.output_path = fname
        self.output_lineedit.setText(self.output_path)

    def make_start_button(self):
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.generate_op)
        self.grid.addWidget(self.start_button, 3, 1)

    def make_param_labels(self):
        self.duration_label = QLabel()
        self.duration_label.setText("Duration: {}".format(self.duration))
        self.param_grid.addWidget(self.duration_label, 1, 0)

        self.frame_count_label = QLabel()
        self.frame_count_label.setText("Frame count: {}".format(self.frame_count))
        self.param_grid.addWidget(self.frame_count_label, 2, 0)

        self.fps_label = QLabel()
        self.fps_label.setText("FPS: {}".format(self.fps))
        self.param_grid.addWidget(self.fps_label, 3, 0)

    def generate_op(self):
        # Generate images from input video
        try:
            generate_video_fames(self.input_path)
            self.duration, self.frame_count, self.fps = get_video_length(self.input_path)
            self.frame_count_label.setText("Frame count: {}".format(self.frame_count))
            self.fps_label.setText("FPS: {}".format(self.fps))
            self.duration_label.setText("Duration: {}".format(self.duration))

        except Exception as e:
            raise e

        # Get files
        files = glob.glob("{}\\*.jpg".format(self.input_path))
        for file in files:
            keypointlist = []
            keypointdict = {}
            try:
                opWrapper = op.WrapperPython()
                opWrapper.configure(params)
                opWrapper.start()
                # Process Image
                datum = op.Datum()
                # datum.fileName=file
                datum.cvInputData = file
                opWrapper.emplaceAndPop([datum])
                # Display Image
                keypointdict['body keypoint'] = np.array(datum.poseKeypoints).tolist()
                keypointlist.append(keypointdict.copy())
                cv2.imshow("Title", datum.cvOutputData)
                cv2.waitKey(1)

            except Exception as e:
                sys.exit(-1)
            filepath = self.output_path
            with open('{}/data.json'.format(filepath), "a") as f:
                json.dump(keypointlist, f, indent=0)


def generate_video_fames(video_path):
    vidcap = cv2.VideoCapture(video_path)

    ''' Make the directory for the images'''
    try:
        if not os.path.exists('input_images'):
            os.makedirs('input_images')
        if not os.path.exists('input_data'):
            os.makedirs('input_data')
    except OSError:
        print('Error: Creating directory of data')

    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = vidcap.read()

        if ret:
            # if video is still left continue creating images
            name = './input_images/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    vidcap.release()
    cv2.destroyAllWindows()


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
    app = QApplication([])
    app.aboutToQuit.connect(app.deleteLater)
    gui = GUI()
    app.exec_()


if __name__ == '__main__':
    main(sys.argv[1:])
