import argparse

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

import support.user_interface as interface
import support.extract_input_data as extract_data
import support.display_data as display_data
import sys


def main(argv=None):
    data = extract_data.ExtractData()
    display = display_data.DisplayData(data)
    # distance_overlay(display, "RBigToe", "LBigToe")
    # display.distance_overlay()
    app = interface.QApplication([])
    app.aboutToQuit.connect(app.deleteLater)
    gui = interface.GUI(display)
    app.exec_()


if __name__ == '__main__':
    main(sys.argv[1:])
