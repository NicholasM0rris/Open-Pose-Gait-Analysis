import os
import sys
import platform
import cv2
import numpy as np
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Import Openpose (Windows)
dir_path = os.path.dirname(os.path.realpath(__file__))
'''
try:
    # Windows Import
    if platform.system() == "Windows":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/python/openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import openpose_python as op
except Exception as e:
    raise e
'''




class GUI(QMainWindow):
    """
    Creates the main GUI window for the user
    """

    def __init__(self):
        super(GUI, self).__init__()
        self.wid = QWidget(self)
        self.grid = QGridLayout()
        self.wid.setLayout(self.grid)
        self.setCentralWidget(self.wid)
        self.setWindowTitle("Early development user interface A204 V2.31")

        self.input()
        self.output()
        self.make_start_button()

        self.show()

    def input(self):
        self.input_label = QLabel()
        self.input_label.setText("Input video: ")
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
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\')
        self.input_path = fname[0]
        self.input_lineedit.setText(self.input_path)

    def open_output(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\')
        self.output_path = fname[0]
        self.output_lineedit.setText(self.output_path)

    def make_start_button(self):
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.generate_op)
        self.grid.addWidget(self.start_button, 3, 1)

    def generate_op(self):
        params = dict()
        params["model_folder"] = "../../../models/"
        params["number_people_max"] = 1

        file = self.input_path
        keypointlist = []
        keypointdict = {}
        try:
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            # Process Image
            datum = op.Datum()
            # datum.fileName=file
            cap = cv2.VideoCapture(file)
            while (cap.isOpened()):
                hasframe, frame = cap.read()
                if hasframe == True:

                    datum.cvInputData = frame
                    opWrapper.emplaceAndPop([datum])

                    # Display Image
                    keypointdict['body keypoint'] = np.array(datum.poseKeypoints).tolist()
                    keypointlist.append(keypointdict.copy())  # must be the copy!!!
                    cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
                    cv2.waitKey(1)
                else:
                    break

        except Exception as e:
            sys.exit(-1)
        filepath = self.output_path

        with open('{}/data.json'.format(filepath), "a") as f:
            json.dump(keypointlist, f, indent=0)



def main(argv=None):

    app = QApplication([])
    app.aboutToQuit.connect(app.deleteLater)
    gui = GUI()
    app.exec_()

if __name__ == '__main__':
    main(sys.argv[1:])
