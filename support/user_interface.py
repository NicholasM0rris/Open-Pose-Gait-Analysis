from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
from os import startfile
import sys
import glob
import support.base_functions as bf


class GUI(QMainWindow):
    """
    Creates the main GUI window for the user
    """

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
        self.setWindowTitle("Early development user interface A204 V2.31")
        self.display = display
        self.display.gui = self
        self.frame_count = self.display.frame_count
        # self.app = QApplication([])
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # self.window = QWidget(parent=self)
        # self.layout = QBoxLayout(QBoxLayout.LeftToRight, self.window)
        """ Create the widgets """
        self.create_tabs()

        self.gif()
        self.gif2()

        self.start_Button()
        self.start_Button2()

        self.dropdown()

        self.print_option_checkbox()
        self.distance_checkbox = Qt.Unchecked
        self.angle_checkbox = Qt.Unchecked
        self.leg_angle_body_checkbox = Qt.Unchecked
        self.left_knee_angle_checkbox = Qt.Unchecked
        self.right_knee_angle_checkbox = Qt.Unchecked

        self.create_step_width_checkbox()
        self.coronal_checkbox = Qt.Unchecked
        self.foot_angle_checkbox = Qt.Unchecked
        self.plot_checkbox()
        self.trajectory_checkbox = Qt.Unchecked

        self.metric_labels()
        self.create_average_step_width_label()

        self.progress_bar()
        self.progress_bar2()

        # self.window.setLayout(self.layout)
        # self.window.show()
        self.tab1.setLayout(self.grid)
        self.tab2.setLayout(self.grid2)
        self.show()

    def create_tabs(self):
        """
        Creates two tabs for sagittal and coronal options
        :return:
        """

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tab.addTab(self.tab1, "Sagittal")
        self.tab.addTab(self.tab2, "Coronal")

    def gif(self):
        """
        Display the walking gif (skeleton currently)
        :return:
        """

        self.movie_label = QLabel()

        self.movie = QMovie("support/skeleton_gif.gif")
        self.movie_label.setMovie(self.movie)

        self.movie.start()
        self.grid.addWidget(self.movie_label, 0, 3)

    def gif2(self):
        """
        Display the walking gif for coronal plane (skeleton currently)
        :return:
        """

        self.movie_label2 = QLabel()

        self.movie2 = QMovie("support/skeleton_walking_coronal.gif")
        self.movie_label2.setMovie(self.movie2)

        self.movie2.start()
        self.grid2.addWidget(self.movie_label2, 0, 3)

    def start_Button(self):
        """
        Create a start button and connect it to start functions
        :return:
        """

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.startbuttonclick)
        self.start_button.clicked.connect(self.start_button_functions)
        self.grid.addWidget(self.start_button, 3, 6)
        # self.start_button.move(300, 400)
        # self.layout.addWidget(self.start_button)

    def start_Button2(self):
        """
        Create a start button for coronal plane tab and connect it to start functions
        :return:
        """
        self.start_button2 = QPushButton('Start', self)
        self.start_button2.clicked.connect(self.startbuttonclick2)
        self.start_button2.clicked.connect(self.start_button_functions2)
        self.grid2.addWidget(self.start_button2, 3, 6)

    @pyqtSlot()
    def start_button_functions2(self):
        """
        Assign worker threads for processing functions in coronal plane
        :return:
        """
        self.worker_thread2 = Worker2(self)
        self.worker_thread2.finish_signal2.connect(self.process_complete_messagebox2)
        self.worker_thread2.start()
        ''' Assign worker signal to function '''
        self.worker_thread2.start_signal2.connect(self.no_option_messagebox)

    @pyqtSlot()
    def start_button_functions(self):
        """
        Assign worker threads for processing functions in the sagittal plane
        :return:
        """
        # Remove any current images in output file
        self.worker_thread = Worker(self)
        self.worker_thread.finish_signal.connect(self.process_complete_messagebox)
        self.worker_thread.start()
        self.worker_thread.start_signal.connect(self.no_option_messagebox)

    @pyqtSlot()
    def no_option_messagebox(self):
        """
        If no options are selected prompt user with a message box
        :return:
        """
        print("No option selected ! ")
        msg = QMessageBox()
        msg.setWindowTitle("Whoops ! ")
        msg.setText("No options were selected ! ")
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_()

    @pyqtSlot()
    def process_complete_messagebox(self):
        """
        Once processing is complete notify the user
        :return:
        """
        print("Process complete ! ")
        msg = QMessageBox.question(self, "The operations have successfully finished ! ",
                                   "Do you want to preview the output video?",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if msg == QMessageBox.Yes:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "processed_video\\Output.avi")
            # print("DIRNAME", path, my_path)
            try:
                startfile(path)
            except FileNotFoundError:
                pass
            try:
                path = os.path.join(my_path, "..\\Output.avi")
                startfile(path)

            except FileNotFoundError:
                print("Preview not supported currently . . . #TODO")
        else:
            pass

    @pyqtSlot()
    def process_complete_messagebox2(self):
        """
        Notify the user with a messagebox once processing in the coronal plane is complete
        :return:
        """
        print("Process complete ! ")
        msg = QMessageBox.question(self, "The operations have successfully finished ! ",
                                   "Do you want to preview the output video?",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if msg == QMessageBox.Yes:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "processed_video\\Coronal_Output.avi")
            # print("DIRNAME", path, my_path)
            startfile(path)
        else:
            pass

    pyqtSlot()

    def startbuttonclick2(self):
        """
        Coronal plane assign progress bar thread
        :return:
        """
        if self.coronal_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.foot_angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if not self.calc:
            self.calc = External2(self)
            self.calc.mySignal2.connect(self.onCountChanged2)
            self.calc.start()

        else:
            print("set counter to 0")
            self.calc.progress = 0
            self.calc.count = 0

    def startbuttonclick(self):
        """
        sagittal plane assign progress bar thread
        :return:
        """
        if self.angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.distance_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.leg_angle_body_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.left_knee_angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if self.right_knee_angle_checkbox == Qt.Checked:
            self.num_operations += 1
        if not self.calc:
            self.calc = External(self)
            self.calc.mySignal.connect(self.onCountChanged)
            self.calc.start()
        else:
            print("set counter to 0")
            self.calc.progress = 0
            self.calc.count = 0

    def dropdown(self):
        """
        Create a drop down to choose which points to calculate distance
        :return:
        """
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
        """
        Check boxes for selecting which measurements to use
        :return:
        """
        self.checkbox_layout = QVBoxLayout()
        label = QLabel("Select your measurements:", self)
        self.checkbox_layout.addWidget(label)

        box = QCheckBox("Distance", self)
        box.stateChanged.connect(self.distance_clickbox)
        self.checkbox_layout.addWidget(box)

        box_angle = QCheckBox("Angle", self)
        box_angle.stateChanged.connect(self.angle_clickbox)
        self.checkbox_layout.addWidget(box_angle)

        leg_body_angle_checkbox = QCheckBox("Leg body angle", self)
        leg_body_angle_checkbox.stateChanged.connect(self.leg_body_angle_clickbox)
        self.checkbox_layout.addWidget(leg_body_angle_checkbox)

        right_knee_angle_checkbox = QCheckBox("Right knee angle", self)
        right_knee_angle_checkbox.stateChanged.connect(self.right_knee_angle_clickbox)
        self.checkbox_layout.addWidget(right_knee_angle_checkbox)

        left_knee_angle_checkbox = QCheckBox("Left knee angle", self)
        left_knee_angle_checkbox.stateChanged.connect(self.left_knee_angle_clickbox)
        self.checkbox_layout.addWidget(left_knee_angle_checkbox)

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

    def create_average_step_width_label(self):
        self.coronal_metrics_layout = QVBoxLayout()
        self.average_step_width_label = QLabel("Average step width: ", self)
        self.coronal_metrics_layout.addWidget(self.average_step_width_label)
        temp_widget = QWidget()
        temp_widget.setLayout(self.coronal_metrics_layout)
        self.grid2.addWidget(temp_widget, 1, 0)

    def create_step_width_checkbox(self):
        self.step_width_layout = QVBoxLayout()
        self.step_width_label = QLabel("Select measurements: ", self)
        self.step_width_label.setAlignment(Qt.AlignBottom)
        self.step_width_layout.addWidget(self.step_width_label)

        box = QCheckBox("Get step width", self)
        box.stateChanged.connect(self.coronal_clickbox)
        foot_angle_box = QCheckBox("Get foot angle")
        foot_angle_box.stateChanged.connect(self.foot_angle_clickbox)
        self.step_width_layout.addWidget(box)
        self.step_width_layout.addWidget(foot_angle_box)

        temp_widget = QWidget()
        temp_widget.setLayout(self.step_width_layout)
        self.grid2.addWidget(temp_widget, 2, 4)

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

    def leg_body_angle_clickbox(self, state):
        if state == Qt.Checked:
            self.leg_angle_body_checkbox = Qt.Checked
            print('leg body Angle Checked')
        else:
            self.leg_angle_body_checkbox = Qt.Unchecked
            print('leg body Angle Unchecked')

    def right_knee_angle_clickbox(self, state):
        if state == Qt.Checked:
            self.right_knee_angle_checkbox = Qt.Checked
            print('right knee Angle Checked')
        else:
            self.right_knee_angle_checkbox = Qt.Unchecked
            print('Right knee Angle Unchecked')

    def left_knee_angle_clickbox(self, state):
        if state == Qt.Checked:
            self.left_knee_angle_checkbox = Qt.Checked
            print('left knee Angle Checked')
        else:
            self.left_knee_angle_checkbox = Qt.Unchecked
            print('left knee Angle Unchecked')

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
            print('Step width Checked')
        else:
            self.coronal_checkbox = Qt.Unchecked
            print('Step width Unchecked')

    def foot_angle_clickbox(self, state):

        if state == Qt.Checked:
            self.foot_angle_checkbox = Qt.Checked
            print('Foot angle Checked')
        else:
            self.foot_angle_checkbox = Qt.Unchecked
            print('Foot angle Unchecked')

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

    @pyqtSlot(int)
    def onCountChanged(self, value):
        self.progress.setValue(value)

    @pyqtSlot(int)
    def onCountChanged2(self, value):
        self.progress2.setValue(value)


TIME_LIMIT = 2400000000000000


class External(QThread):
    """
    run the progress bar in an external thread class
    """
    mySignal = pyqtSignal(int)

    def __init__(self, gui):
        super(External, self).__init__()
        self.gui = gui
        # Set number operations to one to avoid divide by 0 error
        if self.gui.num_operations == 0:
            num_operations = 1
        else:
            num_operations = self.gui.num_operations
        # Get number of files to be operated on
        self.num_files = len(self.gui.display.data.data_files) * num_operations
        self.progress = 0
        self.frame = 1
        self.count = 0

    def run(self):
        try:
            # Get progress to add for each number of file
            add = 100 / self.num_files
        except ZeroDivisionError:
            print("ZeroDivisionError")
            sys.exit()
        print("ADD", add)
        # While not timed out
        while self.count < TIME_LIMIT:
            # Add count
            self.count += 1
            # If the frame number is different to processed frame number
            if self.frame != self.gui.display.frame_number:
                print("Saving to video . . . Progress ({}, {}) / {} frames. {}% complete. {} Frames remaining . . .".format(self.frame, self.gui.display.frame_number, self.gui.frame_count, self.progress, self.gui.frame_count - self.frame))
                # Increase the progress bar and set to new frame
                self.frame = self.gui.display.frame_number
                self.progress += add
                # self.gui.onCountChanged(self.progress)
                self.mySignal.emit(self.progress)
                if self.gui.frame_count - self.frame < 1:
                    print("Process complete . . . Please wait.")
        print("Timer finished")


# TODO fix coronal plane bugs
class External2(QThread):
    """
    Runs progress bar for coronal plane in external thread
    """
    mySignal2 = pyqtSignal(int)

    def __init__(self, gui, parent=None):
        # super(External2, self).__init__()
        QThread.__init__(self, parent)
        self.gui = gui
        if self.gui.num_operations == 0:
            num_operations = 1
        else:
            num_operations = self.gui.num_operations
        self.num_files = len(self.gui.display.data.coronal_data_files) * num_operations
        self.progress = 0
        self.frame = 1
        self.count = 0

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
                # self.gui.onCountChanged2(self.progress)
                print(self.progress)
                self.mySignal2.emit(int(self.progress))


class Worker(QThread):
    """
    Worker thread for processing measurements in the sagittal plane
    """
    start_signal = pyqtSignal()
    finish_signal = pyqtSignal()

    def __init__(self, gui, parent=None):
        # super(External2, self).__init__()
        QThread.__init__(self, parent)
        self.gui = gui

    def run(self):
        files = glob.glob("{}\\*.png".format("output_images"))
        for f in files:
            os.remove(f)
        start = 0
        if self.gui.distance_checkbox == Qt.Checked:
            start = 1
            self.gui.display.distance_overlay()

        if self.gui.angle_checkbox == Qt.Checked:
            start = 1
            self.gui.display.angle_overlay()

        if self.gui.leg_angle_body_checkbox == Qt.Checked:
            start = 1
            self.gui.display.leg_body_angle_overlay()

        if self.gui.left_knee_angle_checkbox == Qt.Checked:
            start = 1
            self.gui.display.left_knee_angle_overlay()

        if self.gui.right_knee_angle_checkbox == Qt.Checked:
            start = 1
            self.gui.display.right_knee_angle_overlay()

        if self.gui.trajectory_checkbox == Qt.Checked:
            start = 1
            self.gui.display.plot_points("RBigToe")

        if start == 0:
            # If no options selected send a signal
            self.start_signal.emit()


        else:
            self.gui.display.display_step_number_overlay()
            for frame in self.gui.display.frame_list:
                bf.save_frame(frame)
            # noinspection PyBroadException
            try:
                bf.save_video()
                self.finish_signal.emit()

            except Exception:
                self.quit()
            # self.process_complete_messagebox()


class Worker2(QThread):
    """
    Worker thread for processing measurements in the coronal plane
    """
    start_signal2 = pyqtSignal()
    finish_signal2 = pyqtSignal()

    def __init__(self, gui, parent=None):
        # super(External2, self).__init__()
        QThread.__init__(self, parent)
        self.gui = gui

    def run(self):
        # Remove any current images in output file
        files = glob.glob("{}\\*.png".format("output_coronal_images"))
        for f in files:
            os.remove(f)
        start = 0

        if self.gui.coronal_checkbox == Qt.Checked:
            start = 1
            self.gui.display.display_step_width()

        if self.gui.foot_angle_checkbox == Qt.Checked:
            start = 1
            self.gui.display.foot_angle_overlay()

        if start == 0:
            self.start_signal2.emit()
        else:
            for frame in self.gui.display.coronal_frame_list:
                bf.save_frame2(frame)
            # noinspection PyBroadException
            try:
                bf.save_video2()
                self.finish_signal2.emit()

            except Exception:
                self.quit()

