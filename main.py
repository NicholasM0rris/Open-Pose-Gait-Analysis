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
ap.add_argument('-a', '--anon', required=False, default=False, help='Set anonymous mode to True or False')
ap.add_argument('-ac', '--anonc', required=False, default=False,
                help='CORONAL PLANE: Set anonymous mode to True or False')
ap.add_argument('-t', '--treadmill', required=False, default=False,
                help='Treadmill: Set treadmill to True if the person is walking on a treadmill')

args = vars(ap.parse_args())

import support.user_interface as interface
import support.extract_input_data as extract_data
import support.display_data as display_data
import updated_interface as test_gui
import sys
import pyqtgraph as pg

app = None


def main(argv=None):
    data = extract_data.ExtractData()
    display = display_data.DisplayData(data)
    # distance_overlay(display, "RBigToe", "LBigToe")
    # display.distance_overlay()
    global app  # This line saved my life. It does not work without it
    app = test_gui.QApplication([])
    # app.aboutToQuit.connect(app.deleteLater) <- bad
    # gui = interface.GUI(display)
    gui = test_gui.GUI(display)
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv[1:])
