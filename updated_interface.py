import os
from os import startfile

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import sys
import glob
import support.base_functions as bf
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.Qt import QUrl



class GUI(QMainWindow):

    def __init__(self, display):
        super(GUI, self).__init__()
        self.num_operations = 0
        self.calc = None
        self.output_movie = ""

        # self.display.gui = self
        # self.frame_count = self.display.frame_count
        # self.app = QApplication([]) (not needed?)
        self.height = 0
        self.pixel_distance = 0
        self.points = []

        self.preview_count = 0
        self.max_preview_count = None

        self.MainWindow = QtWidgets.QMainWindow()
        # ui = GUI(display)
        self.setupUi(self.MainWindow)
        self.MainWindow.setWindowTitle("Early development user interface A204 V2.31")
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        self.display = display
        self.display.gui = self
        self.plot_frame_init()  # TODO currently must be defined after display data
        self.frame_count = self.display.frame_count
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.MainWindow.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(799, 592)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 801, 581))
        self.tabWidget.setObjectName("tabWidget")

        self.init_calibrate_tab()
        self.init_sagittal_tab()
        self.coronal_tab_init()
        self.metrics_tab_init()
        self.plots_tab_init()

    def init_calibrate_tab(self):
        self.calibrate_tab = QtWidgets.QWidget()
        self.calibrate_tab.setObjectName("calibrate_tab")

        self.init_height_widgets()
        self.init_weight_widgets()
        self.init_calibrate_frame()

    def init_height_widgets(self):
        self.onlyInt = QIntValidator()
        self.height_lineEdit = QtWidgets.QLineEdit(self.calibrate_tab)
        self.height_lineEdit.setGeometry(QtCore.QRect(170, 436, 113, 20))
        self.height_lineEdit.setObjectName("height_lineEdit")
        self.height_lineEdit.setValidator(self.onlyInt)
        self.height_label = QtWidgets.QLabel(self.calibrate_tab)
        self.height_label.setGeometry(QtCore.QRect(4, 428, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.height_label.setFont(font)
        self.height_label.setObjectName("height_label")
        self.calibrate_height_pushButton = QtWidgets.QPushButton(self.calibrate_tab)
        self.calibrate_height_pushButton.setGeometry(QtCore.QRect(282, 436, 75, 21))
        self.calibrate_height_pushButton.setObjectName("calibrate_height_pushButton")
        self.calibrate_height_pushButton.clicked.connect(self.calibrateheightbuttonclick)

        self.ratio_label = QtWidgets.QLabel(self.calibrate_tab)
        self.ratio_label.setGeometry(QtCore.QRect(474, 452, 305, 43))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.ratio_label.setFont(font)
        self.ratio_label.setObjectName("ratio_label")

    def pixels_to_mm(self):
        self.pixel_ratio = self.height / self.pixel_distance
        self.ratio_label.setText("Pixel to millimetre ratio: {}".format(str(self.pixel_ratio)))
        print("One pixel is equivalent to {} mms".format(self.pixel_ratio))

    def calibrateheightbuttonclick(self):
        try:
            self.height = int(self.height_lineEdit.text())
            if self.display.image_path:
                self.calibrate_thread = Worker3(self)
                # self.calibrate_thread.start()
            else:
                print("Select an image!")
        except ValueError:
            print("Enter a height first")

    def init_weight_widgets(self):
        self.weight_label_2 = QtWidgets.QLabel(self.calibrate_tab)
        self.weight_label_2.setGeometry(QtCore.QRect(362, 476, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.weight_label_2.setFont(font)
        self.weight_label_2.setObjectName("weight_label_2")
        self.weight_label_3 = QtWidgets.QLabel(self.calibrate_tab)
        self.weight_label_3.setGeometry(QtCore.QRect(4, 474, 165, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.weight_label_3.setFont(font)
        self.weight_label_3.setObjectName("weight_label_3")
        self.weight_lineEdit_2 = QtWidgets.QLineEdit(self.calibrate_tab)
        self.weight_lineEdit_2.setGeometry(QtCore.QRect(170, 484, 113, 20))
        self.weight_lineEdit_2.setObjectName("weight_lineEdit_2")
        self.weight_pushButton_2 = QtWidgets.QPushButton(self.calibrate_tab)
        self.weight_pushButton_2.setGeometry(QtCore.QRect(282, 484, 75, 21))
        self.weight_pushButton_2.setObjectName("weight_pushButton_2")

    def init_calibrate_frame(self):
        self.image_label = QLabel(self.calibrate_tab)
        self.image_label.setGeometry(QtCore.QRect(128, 60, 543, 353))

        # self.image = QMovie("support/skeleton_gif.gif")

        # self.frame = QtWidgets.QFrame(self.calibrate_tab)
        # self.frame.setGeometry(QtCore.QRect(128, 60, 543, 353))
        # self.frame.setAutoFillBackground(True)
        # self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        # self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        # self.frame.setObjectName("frame")

        self.calibrate_frame_lineEdit_2 = QtWidgets.QLineEdit(self.calibrate_tab)
        self.calibrate_frame_lineEdit_2.setGeometry(QtCore.QRect(130, 22, 263, 20))
        self.calibrate_frame_lineEdit_2.setObjectName("calibrate_frame_lineEdit_2")
        # The enter button
        self.calibrate_frame_pushButton_2 = QtWidgets.QPushButton(self.calibrate_tab)
        self.calibrate_frame_pushButton_2.setGeometry(QtCore.QRect(394, 22, 75, 21))
        self.calibrate_frame_pushButton_2.setObjectName("calibrate_frame_pushButton_2")

        self.calibrate_browsefile_pushButton = QtWidgets.QPushButton(self.calibrate_tab)
        self.calibrate_browsefile_pushButton.setGeometry(QtCore.QRect(468, 22, 75, 21))
        self.calibrate_browsefile_pushButton.setObjectName("calibrate_browsefile_pushButton")
        self.calibrate_browsefile_pushButton.clicked.connect(self.calibrate_open_file)

        self.calibrate_selectframe_label_2 = QtWidgets.QLabel(self.calibrate_tab)
        self.calibrate_selectframe_label_2.setGeometry(QtCore.QRect(12, 14, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.calibrate_selectframe_label_2.setFont(font)
        self.calibrate_selectframe_label_2.setObjectName("calibrate_selectframe_label_2")
        self.tabWidget.addTab(self.calibrate_tab, "")

    def calibrate_open_file(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file')
        self.calibrate_file_path = str(fname[0])
        self.calibrate_frame_lineEdit_2.setText(self.calibrate_file_path)
        self.display.image_path = self.calibrate_file_path

    def init_sagittal_tab(self):
        self.saggital_tab = QtWidgets.QWidget()
        self.saggital_tab.setObjectName("saggital_tab")

        self.s_video_output_path = ""

        self.init_s_vid_frame()
        self.init_s_start_button()
        self.init_s_video()
        self.s_init_checkboxes()
        self.s_init_progressbar()
        self.s_init_save_output()
        self.s_init_dropdownbox()

    def init_s_vid_frame(self):
        self.s_vid_frame = QLabel(self.saggital_tab)

        # self.s_vid = QMovie("support/skeleton_gif.gif")
        self.s_vid_frame.setGeometry(QtCore.QRect(178, 30, 409, 311))
        # self.s_vid_frame.setMovie(self.s_vid)
        # self.s_vid.start()
        example_image = 'example1.png'

        self.s_vid_frame.setObjectName("s_vid_frame")
        pixmap = QPixmap(example_image)
        self.s_vid_frame.setPixmap(pixmap)
        # This method is very slow!!!!!

        self.video = QVideoWidget(self.saggital_tab)
        self.video.setGeometry(QtCore.QRect(178, 30, 409, 311))
        self.video.setObjectName("video")

        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.positionChanged.connect(self.positionChanged)
        # self.player.stateChanged.connect(self.mediaStateChanged)

        # TODO slider not working as expected: removed for other features

        self.positionSlider = QSlider(orientation=Qt.Horizontal, parent=self.saggital_tab)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        self.positionSlider.setRange(0, 30000)
        self.positionSlider.setGeometry(178, 352, 409, 22)
        self.test_var = None

        # file = os.path.join(os.path.dirname(__file__), "small.mp4")
        '''
        self.s_vid_frame = QtWidgets.QFrame(self.saggital_tab)
        self.s_vid_frame.setGeometry(QtCore.QRect(178, 30, 409, 311))
        self.s_vid_frame.setAutoFillBackground(True)
        self.s_vid_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.s_vid_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.s_vid_frame.setObjectName("s_vid_frame")
        '''

    @pyqtSlot(int)
    def setPosition(self, position):
        print('set position', position, type(position))
        self.player.setPosition(int(position))

    @pyqtSlot('qint64')
    def positionChanged(self, position):
        # print('position changed', position, type(position))
        # print("set duration to ", self.player.duration())
        # self.positionSlider.setRange(0, self.player.duration())
        self.positionSlider.setValue(position)

    '''
        @pyqtSlot('qint64')
        def durationChanged(self, duration):
            print('duration changed', duration, type(duration))
            self.positionSlider.setRange(0, duration)

        @pyqtSlot()
        def mediaStatusChanged(self):
            print("set duration to ", self.player.duration())
            self.positionSlider.setRange(0, self.player.duration())

        @pyqtSlot()
        def mediaStateChanged(self):
            print("set duration to ", self.player.duration())
            self.positionSlider.setRange(0, self.player.duration())
    '''

    def init_s_start_button(self):
        self.s_start_pushButton_3 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_start_pushButton_3.setGeometry(QtCore.QRect(680, 478, 89, 41))
        self.s_start_pushButton_3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_start_pushButton_3.setObjectName("s_start_pushButton_3")
        self.s_start_pushButton_3.clicked.connect(self.startbuttonclick)
        self.s_start_pushButton_3.clicked.connect(self.start_button_functions)

    def startbuttonclick(self):
        """
        sagittal plane assign progress bar thread
        :return:
        """
        if self.s_angle_checkBox == Qt.Checked:
            self.num_operations += 1
        if self.s_distance_checkBox_6 == Qt.Checked:
            self.num_operations += 1
        if self.s_legangle_checkBox_4 == Qt.Checked:
            self.num_operations += 1
        if self.s_lknee_angle_checkBox_2 == Qt.Checked:
            self.num_operations += 1
        if self.s_rknee_angle_checkBox_3 == Qt.Checked:
            self.num_operations += 1
        if not self.calc:

            self.calc = External(self)
            self.calc.mySignal.connect(self.s_onCountChanged)
            # self.calc.start()

        else:
            print("set counter to 0")
            self.calc.progress = 0
            self.calc.count = 0

    @pyqtSlot()
    def start_button_functions(self):
        """
        Assign worker threads for processing functions in the sagittal plane
        """
        # Remove any current images in output file

        self.worker_thread = Worker(self)
        self.worker_thread.finish_signal.connect(self.process_complete_messagebox)
        # self.worker_thread.start()
        self.worker_thread.start_signal.connect(self.no_option_messagebox)

    @pyqtSlot()
    def no_option_messagebox(self):
        """
        If no options are selected prompt user with a message box
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
        """
        if self.s_video_output_path == "":
            try:
                self.player.setMedia(QMediaContent(QUrl.fromLocalFile('Output.avi')))
                self.player.setVideoOutput(self.video)
            except Exception as e:
                raise e

        else:
            try:
                self.player.setMedia(QMediaContent(QUrl.fromLocalFile('{}/Output.avi'.format(self.s_video_output_path))))
                self.player.setVideoOutput(self.video)
            # DirectShowPlayerService error (The video doesn't exist so just pass)
            except Exception as e:
                pass
        # self.player.mediaStatusChanged.connect(self.mediaStatusChanged)

        # self.player.durationChanged.connect(self.durationChanged)
        # self.player.stateChanged.connect(self.mediaStateChanged)

        print('dr', self.player.duration())
        self.player.play()
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
                path = os.path.join(my_path, "Output.avi")
                startfile(path)

            except FileNotFoundError:
                print("Preview not supported currently . . . #TODO")
        else:
            pass

    @pyqtSlot()
    def coronal_process_complete_messagebox(self):
        """
        Once processing is complete notify the user
        """
        if self.c_video_output_path == "":
            try:
                print("err1")
                self.coronal_player.setMedia(
                    QMediaContent(QUrl.fromLocalFile('Output.avi')))
                self.coronal_player.setVideoOutput(self.coronal_video)
            except Exception as e:
                raise e

        else:
            try:
                print("eer2")
                self.coronal_player.setMedia(
                    QMediaContent(QUrl.fromLocalFile('{}/Output.avi'.format(self.c_video_output_path))))
                self.coronal_player.setVideoOutput(self.coronal_video)
            # DirectShowPlayerService error (The video doesn't exist so just pass)
            except Exception as e:
                pass

        # self.player.mediaStatusChanged.connect(self.mediaStatusChanged)

        # self.player.durationChanged.connect(self.durationChanged)
        # self.player.stateChanged.connect(self.mediaStateChanged)

        self.coronal_player.play()
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
                path = os.path.join(my_path, "Output.avi")
                startfile(path)

            except FileNotFoundError:
                print("Preview not supported currently . . . #TODO")
        else:
            pass

    def s_open_vid(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Movie",
                                                            QtCore.QDir.homePath())
        if fileName:
            self.player.setMedia(
                QMediaContent(QtCore.QUrl.fromLocalFile(fileName)))
            self.player.setVideoOutput(self.video)

    def init_s_video(self):

        self.s_vid_open_pushButton_6 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_vid_open_pushButton_6.setGeometry(QtCore.QRect(620, 202, 131, 43))
        self.s_vid_open_pushButton_6.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_vid_open_pushButton_6.setObjectName("s_vid_open_pushButton_6")
        self.s_vid_open_pushButton_6.clicked.connect(self.s_open_vid)

        self.s_play_pushButton_7 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_play_pushButton_7.setGeometry(QtCore.QRect(620, 62, 131, 43))
        self.s_play_pushButton_7.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_play_pushButton_7.setObjectName("s_play_pushButton_7")
        self.s_play_pushButton_7.clicked.connect(self.s_play_pushButton_function)

        self.s_back_pushButton_8 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_back_pushButton_8.setGeometry(QtCore.QRect(620, 132, 65, 43))
        self.s_back_pushButton_8.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_back_pushButton_8.setObjectName("s_back_pushButton_8")
        self.s_back_pushButton_8.clicked.connect(self.s_backbutton_video_function)

        self.s_forward_pushButton_9 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_forward_pushButton_9.setGeometry(QtCore.QRect(684, 132, 67, 43))
        self.s_forward_pushButton_9.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_forward_pushButton_9.setObjectName("s_forward_pushButton_9")
        self.s_forward_pushButton_9.clicked.connect(self.s_forwardbutton_video_function)

        self.s_measurements_label_6 = QtWidgets.QLabel(self.saggital_tab)
        self.s_measurements_label_6.setGeometry(QtCore.QRect(4, 32, 169, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.s_measurements_label_6.setFont(font)
        self.s_measurements_label_6.setObjectName("s_measurements_label_6")
        self.s_playback_label_7 = QtWidgets.QLabel(self.saggital_tab)
        self.s_playback_label_7.setGeometry(QtCore.QRect(616, 32, 169, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.s_playback_label_7.setFont(font)
        self.s_playback_label_7.setObjectName("s_playback_label_7")
        self.tabWidget.addTab(self.saggital_tab, "")

    def s_backbutton_video_function(self, position):
        # print('set position', position, type(position))

        self.player.setPosition(int(self.player.position() - 3000))

    def s_forwardbutton_video_function(self, position):
        # print('set position', position, type(position))
        # print(self.player.position(), self.player.position)
        self.player.setPosition(int(self.player.position() + 3000))

    def s_play_pushButton_function(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def s_backbutton_function(self):
        if self.max_preview_count:
            if self.preview_count - 1 < 1:
                self.preview_count = self.max_preview_count
            else:
                self.preview_count -= 1
            img_path = 'display_images/{}.png'.format(self.preview_count)
            pixmap = QPixmap(img_path)
            self.s_vid_frame.setPixmap(pixmap)

        else:
            pass

    def s_forwardbutton_function(self):
        if self.max_preview_count:
            if self.preview_count + 1 > self.max_preview_count:
                self.preview_count = 1
            else:
                self.preview_count += 1
            img_path = 'display_images/{}.png'.format(self.preview_count)
            pixmap = QPixmap(img_path)
            self.s_vid_frame.setPixmap(pixmap)

        else:
            pass

    def s_init_checkboxes(self):
        self.s_angle_checkBox = QtWidgets.QCheckBox(self.saggital_tab)
        self.s_angle_checkBox.setGeometry(QtCore.QRect(8, 60, 149, 35))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_angle_checkBox.setFont(font)
        self.s_angle_checkBox.setToolTipDuration(-1)
        self.s_angle_checkBox.setObjectName("s_angle_checkBox")
        self.s_angle_checkBox.stateChanged.connect(self.angle_clickbox_function)
        self.s_lknee_angle_checkBox_2 = QtWidgets.QCheckBox(self.saggital_tab)
        self.s_lknee_angle_checkBox_2.setGeometry(QtCore.QRect(8, 178, 149, 35))
        self.s_lknee_angle_checkBox_2.stateChanged.connect(self.left_knee_angle_clickbox_function)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_lknee_angle_checkBox_2.setFont(font)
        self.s_lknee_angle_checkBox_2.setObjectName("s_lknee_angle_checkBox_2")
        self.s_rknee_angle_checkBox_3 = QtWidgets.QCheckBox(self.saggital_tab)
        self.s_rknee_angle_checkBox_3.setGeometry(QtCore.QRect(8, 138, 149, 35))
        self.s_rknee_angle_checkBox_3.stateChanged.connect(self.right_knee_angle_clickbox_function)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_rknee_angle_checkBox_3.setFont(font)
        self.s_rknee_angle_checkBox_3.setObjectName("s_rknee_angle_checkBox_3")
        self.s_legangle_checkBox_4 = QtWidgets.QCheckBox(self.saggital_tab)
        self.s_legangle_checkBox_4.setGeometry(QtCore.QRect(8, 100, 149, 35))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_legangle_checkBox_4.setFont(font)
        self.s_legangle_checkBox_4.setObjectName("s_legangle_checkBox_4")
        self.s_legangle_checkBox_4.stateChanged.connect(self.leg_body_angle_clickbox_function)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_distance_checkBox_6 = QtWidgets.QCheckBox(self.saggital_tab)
        self.s_distance_checkBox_6.setGeometry(QtCore.QRect(12, 356, 93, 35))
        self.s_distance_checkBox_6.stateChanged.connect(self.distance_clickbox_function)
        self.s_distance_checkBox_6.setFont(font)
        self.s_distance_checkBox_6.setObjectName("s_distance_checkBox_6")

    def leg_body_angle_clickbox_function(self, state):
        if state == Qt.Checked:
            self.s_legangle_checkBox_4 = Qt.Checked
            print('leg body Angle Checked')
        else:
            self.s_legangle_checkBox_4 = Qt.Unchecked
            print('leg body Angle Unchecked')

    def distance_clickbox_function(self, state):
        if state == Qt.Checked:
            self.s_distance_checkBox_6 = Qt.Checked
            print('Distance Checked')
        else:
            self.s_distance_checkBox_6 = Qt.Unchecked
            print('Distance Unchecked')

    def right_knee_angle_clickbox_function(self, state):
        if state == Qt.Checked:
            self.s_rknee_angle_checkBox_3 = Qt.Checked
            print('right knee Angle Checked')
        else:
            self.s_rknee_angle_checkBox_3 = Qt.Unchecked
            print('Right knee Angle Unchecked')

    def angle_clickbox_function(self, state):
        if state == Qt.Checked:
            self.s_angle_checkBox = Qt.Checked
            print('Angle Checked')
        else:
            self.s_angle_checkBox = Qt.Unchecked
            print('Angle Unchecked')

    def left_knee_angle_clickbox_function(self, state):
        if state == Qt.Checked:
            self.s_lknee_angle_checkBox_2 = Qt.Checked
            print('left knee Angle Checked')
        else:
            self.s_lknee_angle_checkBox_2 = Qt.Unchecked
            print('left knee Angle Unchecked')

    def s_init_progressbar(self):
        self.s_progressBar = QtWidgets.QProgressBar(self.saggital_tab)
        self.s_progressBar.setGeometry(QtCore.QRect(14, 480, 661, 37))
        # self.s_progressBar.setCursor(QtGui.QCursor(QtCore.Qt.BusyCursor))
        self.s_progressBar.setProperty("value", 0)
        self.s_progressBar.setMaximum(100)
        self.s_progressBar.setObjectName("s_progressBar")

    @pyqtSlot(int)
    def s_onCountChanged(self, value):

        self.s_progressBar.setValue(value)

    def s_init_save_output(self):
        self.s_saveoutput_checkBox_5 = QtWidgets.QCheckBox(self.saggital_tab)
        self.s_saveoutput_checkBox_5.setGeometry(QtCore.QRect(476, 398, 149, 35))
        self.s_saveoutput_checkBox_5.stateChanged.connect(self.s_saveoutput_clickbox_function)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_saveoutput_checkBox_5.setFont(font)
        self.s_saveoutput_checkBox_5.setObjectName("s_saveoutput_checkBox_5")
        self.s_saveoutput_lineEdit_3 = QtWidgets.QLineEdit(self.saggital_tab)
        self.s_saveoutput_lineEdit_3.setGeometry(QtCore.QRect(476, 436, 233, 29))
        self.s_saveoutput_lineEdit_3.setObjectName("s_saveoutput_lineEdit_3")

        self.s_saveoutput_browse_pushButton = QtWidgets.QPushButton(self.saggital_tab)
        self.s_saveoutput_browse_pushButton.setGeometry(QtCore.QRect(712, 436, 77, 29))
        self.s_saveoutput_browse_pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_saveoutput_browse_pushButton.setObjectName("s_saveoutput_browse_pushButton")
        self.s_saveoutput_browse_pushButton.clicked.connect(self.saggital_open_output_directory)

    def saggital_open_output_directory(self):

        dir = QFileDialog.getExistingDirectory(self, 'Select output directory')
        self.s_video_output_path = str(dir)
        self.s_saveoutput_lineEdit_3.setText(self.s_video_output_path)

    def s_saveoutput_clickbox_function(self, state):
        if state == Qt.Checked:
            self.s_saveoutput_checkBox_5 = Qt.Checked
            print('S save output Checked')
        else:
            self.s_saveoutput_checkBox_5 = Qt.Unchecked
            print('S save output Unchecked')

    def set_dropdown1(self, text):
        self.display.keypoint1 = text
        print(self.display.keypoint1)

    def set_dropdown2(self, text):
        self.display.keypoint2 = text
        print(self.display.keypoint2)

    def s_init_dropdownbox(self):
        self.s_dp2_comboBox = QtWidgets.QComboBox(self.saggital_tab)
        self.s_dp2_comboBox.setGeometry(QtCore.QRect(246, 428, 217, 39))
        self.s_dp2_comboBox.setObjectName("s_dp2_comboBox")
        self.s_dp2_comboBox.addItem("RBigToe")
        self.s_dp2_comboBox.addItem("RWrist")
        self.s_dp2_comboBox.addItem("RElbow")
        self.s_dp2_comboBox.addItem("REye")
        self.s_dp2_comboBox.addItem("RHeel")
        self.s_dp2_comboBox.addItem("RAnkle")
        self.s_dp2_comboBox.addItem("RHip")
        self.s_dp2_comboBox.addItem("REar")
        self.s_dp2_comboBox.addItem("RShoulder")
        self.s_dp2_comboBox.addItem("MidHip")
        self.s_dp2_comboBox.addItem("Nose")
        self.s_dp2_comboBox.addItem("Neck")
        self.s_dp2_comboBox.activated[str].connect(self.set_dropdown2)

        self.s_dp1_comboBox_3 = QtWidgets.QComboBox(self.saggital_tab)
        self.s_dp1_comboBox_3.setGeometry(QtCore.QRect(14, 428, 217, 39))
        self.s_dp1_comboBox_3.setObjectName("s_dp1_comboBox_3")
        self.s_dp1_comboBox_3.addItem("LBigToe")
        self.s_dp1_comboBox_3.addItem("LWrist")
        self.s_dp1_comboBox_3.addItem("LElbow")
        self.s_dp1_comboBox_3.addItem("LEye")
        self.s_dp1_comboBox_3.addItem("LHeel")
        self.s_dp1_comboBox_3.addItem("LAnkle")
        self.s_dp1_comboBox_3.addItem("LHip")
        self.s_dp1_comboBox_3.addItem("LEar")
        self.s_dp1_comboBox_3.addItem("LShoulder")
        self.s_dp1_comboBox_3.addItem("MidHip")
        self.s_dp1_comboBox_3.addItem("Nose")
        self.s_dp1_comboBox_3.addItem("Neck")
        self.s_dp1_comboBox_3.activated[str].connect(self.set_dropdown2)

        self.s_dp1_label_4 = QtWidgets.QLabel(self.saggital_tab)
        self.s_dp1_label_4.setGeometry(QtCore.QRect(16, 400, 165, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_dp1_label_4.setFont(font)
        self.s_dp1_label_4.setObjectName("s_dp1_label_4")
        self.s_dp2_label_5 = QtWidgets.QLabel(self.saggital_tab)
        self.s_dp2_label_5.setGeometry(QtCore.QRect(246, 400, 165, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_dp2_label_5.setFont(font)
        self.s_dp2_label_5.setObjectName("s_dp2_label_5")

    def coronal_tab_init(self):
        self.coronal_tab = QtWidgets.QWidget()
        self.coronal_tab.setObjectName("coronal_tab")

        self.s_video_output_path = ""
        self.c_video_output_path = ""

        self.c_save_output_init()
        self.c_progressbar_init()
        self.c_startbutton_init()
        self.c_measurement_checkboxes_init()
        self.c_frame_init()

        self.tabWidget.addTab(self.coronal_tab, "")

    def c_save_output_init(self):
        self.c_saveoutput_pushButton = QtWidgets.QPushButton(self.coronal_tab)
        self.c_saveoutput_pushButton.setGeometry(QtCore.QRect(712, 436, 77, 29))
        self.c_saveoutput_pushButton.setObjectName("c_saveoutput_pushButton")
        self.c_saveoutput_pushButton.clicked.connect(self.coronal_open_output_directory)

        self.c_output_lineEdit = QtWidgets.QLineEdit(self.coronal_tab)
        self.c_output_lineEdit.setGeometry(QtCore.QRect(476, 436, 233, 29))
        self.c_output_lineEdit.setObjectName("c_output_lineEdit")

    def coronal_open_output_directory(self):

        dir = QFileDialog.getExistingDirectory(self, 'Select output directory')
        self.c_video_output_path = str(dir)
        self.c_output_lineEdit.setText(self.c_video_output_path)

    def c_backbutton_video_function(self, position):
        # print('set position', position, type(position))

        self.coronal_player.setPosition(int(self.coronal_player.position() - 3000))

    def c_forwardbutton_video_function(self, position):
        # print('set position', position, type(position))
        # print(self.player.position(), self.player.position)
        self.coronal_player.setPosition(int(self.coronal_player.position() + 3000))

    def c_play_pushButton_function(self):
        if self.coronal_player.state() == QMediaPlayer.PlayingState:
            self.coronal_player.pause()
        else:
            self.coronal_player.play()
            self.c_vid_frame.hide()

    def c_progressbar_init(self):
        self.progressBar_2 = QtWidgets.QProgressBar(self.coronal_tab)
        self.progressBar_2.setGeometry(QtCore.QRect(14, 480, 661, 37))
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")

    def c_startbutton_init(self):
        self.c_start_pushButton = QtWidgets.QPushButton(self.coronal_tab)
        self.c_start_pushButton.setGeometry(QtCore.QRect(680, 478, 89, 41))
        self.c_start_pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.c_start_pushButton.setObjectName("c_start_pushButton")
        self.c_start_pushButton.clicked.connect(self.coronal_startbuttonclick)
        self.c_start_pushButton.clicked.connect(self.coronal_start_button_functions)

    @pyqtSlot(int)
    def c_onCountChanged(self, value):

        self.progressBar_2.setValue(value)

    def coronal_startbuttonclick(self):
        """
        sagittal plane assign progress bar thread
        :return:
        """

        if self.c_stepwidth_checkBox == Qt.Checked:
            self.num_operations += 1
        if self.c_footangle_checkBox == Qt.Checked:
            self.num_operations += 1
        if not self.calc:
            self.calc = CoronalExternal(self)
            self.calc.mySignal.connect(self.c_onCountChanged)
            # self.calc.start()
        else:
            print("set counter to 0")
            self.calc.progress = 0
            self.calc.count = 0

    @pyqtSlot()
    def coronal_start_button_functions(self):
        """
        Assign worker threads for processing functions in the sagittal plane
        """
        # Remove any current images in output file

        self.coronal_worker_thread = CoronalWorker(self)
        self.coronal_worker_thread.finish_signal.connect(self.coronal_process_complete_messagebox)
        # self.coronal_worker_thread.start()
        self.coronal_worker_thread.start_signal.connect(self.no_option_messagebox)

    def c_stepwidth_clickbox(self, state):
        if state == Qt.Checked:
            self.c_stepwidth_checkBox = Qt.Checked
            print('Step width Checked')
        else:
            self.c_stepwidth_checkBox = Qt.Unchecked
            print('Step width Unchecked')

    def c_foot_angle_clickbox(self, state):

        if state == Qt.Checked:
            self.c_footangle_checkBox = Qt.Checked
            print('Foot angle Checked')
        else:
            self.c_footangle_checkBox = Qt.Unchecked
            print('Foot angle Unchecked')

    def c_measurement_checkboxes_init(self):
        self.c_footangle_checkBox = QtWidgets.QCheckBox(self.coronal_tab)
        self.c_footangle_checkBox.setGeometry(QtCore.QRect(22, 418, 167, 41))
        self.c_footangle_checkBox.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.c_footangle_checkBox.setFont(font)
        self.c_footangle_checkBox.setIconSize(QtCore.QSize(16, 16))
        self.c_footangle_checkBox.setObjectName("c_footangle_checkBox")
        self.c_footangle_checkBox.stateChanged.connect(self.c_foot_angle_clickbox)

        self.c_stepwidth_checkBox = QtWidgets.QCheckBox(self.coronal_tab)
        self.c_stepwidth_checkBox.setGeometry(QtCore.QRect(22, 366, 167, 53))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.c_stepwidth_checkBox.setFont(font)
        self.c_stepwidth_checkBox.setToolTipDuration(-1)
        self.c_stepwidth_checkBox.setStatusTip("")
        self.c_stepwidth_checkBox.setWhatsThis("")
        self.c_stepwidth_checkBox.setObjectName("c_stepwidth_checkBox")
        self.c_stepwidth_checkBox.stateChanged.connect(self.c_stepwidth_clickbox)

        self.c_measurements_label = QtWidgets.QLabel(self.coronal_tab)
        self.c_measurements_label.setGeometry(QtCore.QRect(5, 350, 169, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.c_measurements_label.setFont(font)
        self.c_measurements_label.setObjectName("c_measurements_label")

    def coronal_positionChanged(self, position):
        self.coronal_positionSlider.setValue(position)

    def coronal_setPosition(self, position):
        self.coronal_player.setPosition(int(position))

    def c_frame_init(self):
        '''
        self.frame_2 = QtWidgets.QFrame(self.coronal_tab)
        self.frame_2.setGeometry(QtCore.QRect(178, 30, 409, 311))
        self.frame_2.setAutoFillBackground(True)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        '''

        self.coronal_video = QVideoWidget(self.coronal_tab)
        self.coronal_video.setGeometry(QtCore.QRect(178, 30, 409, 311))
        self.coronal_video.setObjectName("c_video")

        self.coronal_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.coronal_player.positionChanged.connect(self.coronal_positionChanged)

        self.coronal_positionSlider = QSlider(orientation=Qt.Horizontal, parent=self.coronal_tab)
        self.coronal_positionSlider.sliderMoved.connect(self.coronal_setPosition)
        self.coronal_positionSlider.setRange(0, 30000)
        self.coronal_positionSlider.setGeometry(178, 352, 409, 22)

        self.c_saveoutput_checkBox = QtWidgets.QCheckBox(self.coronal_tab)
        self.c_saveoutput_checkBox.setGeometry(QtCore.QRect(476, 398, 149, 35))

        font = QtGui.QFont()
        font.setPointSize(12)
        self.c_saveoutput_checkBox.setFont(font)
        self.c_saveoutput_checkBox.setObjectName("c_saveoutput_checkBox")
        self.c_saveoutput_checkBox.stateChanged.connect(self.c_saveoutput_clickbox_function)

        self.c_forward_pushButton = QtWidgets.QPushButton(self.coronal_tab)
        self.c_forward_pushButton.setGeometry(QtCore.QRect(684, 132, 67, 43))
        self.c_forward_pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.c_forward_pushButton.setObjectName("c_forward_pushButton")
        self.c_forward_pushButton.clicked.connect(self.c_forwardbutton_video_function)

        self.c_openfile_pushButton = QtWidgets.QPushButton(self.coronal_tab)
        self.c_openfile_pushButton.setGeometry(QtCore.QRect(620, 202, 131, 43))
        self.c_openfile_pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.c_openfile_pushButton.setObjectName("c_openfile_pushButton")
        self.c_openfile_pushButton.clicked.connect(self.c_open_vid)

        self.c_back_pushButton = QtWidgets.QPushButton(self.coronal_tab)
        self.c_back_pushButton.setGeometry(QtCore.QRect(620, 132, 65, 43))
        self.c_back_pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.c_back_pushButton.setObjectName("c_back_pushButton")
        self.c_back_pushButton.clicked.connect(self.c_backbutton_video_function)

        self.c_playback_label = QtWidgets.QLabel(self.coronal_tab)
        self.c_playback_label.setGeometry(QtCore.QRect(616, 32, 169, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.c_playback_label.setFont(font)
        self.c_playback_label.setObjectName("c_playback_label")
        self.c_playback_pushButton = QtWidgets.QPushButton(self.coronal_tab)
        self.c_playback_pushButton.setGeometry(QtCore.QRect(620, 62, 131, 43))
        self.c_playback_pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.c_playback_pushButton.setObjectName("c_playback_pushButton")
        self.c_playback_pushButton.clicked.connect(self.c_play_pushButton_function)

        self.c_vid_frame = QLabel(self.coronal_tab)
        self.c_vid_frame.setGeometry(QtCore.QRect(178, 30, 409, 311))
        example_image = 'example1.png'
        self.c_vid_frame.setObjectName("c_vid_frame")
        pixmap = QPixmap(example_image)
        self.c_vid_frame.setPixmap(pixmap)

    def c_open_vid(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Movie",
                                                            QtCore.QDir.homePath())
        if fileName:
            self.coronal_player.setMedia(
                QMediaContent(QtCore.QUrl.fromLocalFile(fileName)))
            self.coronal_player.setVideoOutput(self.coronal_video)

    def c_saveoutput_clickbox_function(self, state):
        if state == Qt.Checked:
            self.c_saveoutput_checkBox = Qt.Checked
            print('C save output Checked')
        else:
            self.c_saveoutput_checkBox = Qt.Unchecked
            print('C save output Unchecked')

    def metrics_tab_init(self):
        self.metrics_tab = QtWidgets.QWidget()
        self.metrics_tab.setObjectName("metrics_tab")
        self.tabWidget.addTab(self.metrics_tab, "")

    def plots_tab_init(self):
        self.plots_tab = QtWidgets.QWidget()
        self.plots_tab.setObjectName("plots_tab")
        self.plot_dropdown_init()

    def plot_frame_init(self):

        ''' Filtered cadence scatter plot '''
        self.filtered_cadence_scatter_plot = self.display.filtered_cadence_scatter_plot
        self.filtered_cadence_scatter_toolbar = NavigationToolbar(self.filtered_cadence_scatter_plot, self)
        self.filtered_cadence_scatter_plot_layout = QtWidgets.QVBoxLayout()
        self.filtered_cadence_scatter_plot_layout.addWidget(self.filtered_cadence_scatter_toolbar)
        self.filtered_cadence_scatter_plot_layout.addWidget(self.display.filtered_cadence_scatter_plot)

        self.filtered_cadence_scatter_plot_widget = QtWidgets.QWidget(self.plots_tab)
        self.filtered_cadence_scatter_plot_widget.setLayout(self.filtered_cadence_scatter_plot_layout)
        self.filtered_cadence_scatter_plot_widget.setGeometry(QtCore.QRect(90, 10, 619, 403))
        self.filtered_cadence_scatter_plot_widget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        ''' Filtered cadence line plot '''
        self.filtered_cadence_line_toolbar = NavigationToolbar(self.display.filtered_cadence_line_plot, self)
        self.filtered_cadence_line_plot_layout = QtWidgets.QVBoxLayout()
        self.filtered_cadence_line_plot_layout.addWidget(self.filtered_cadence_line_toolbar)
        self.filtered_cadence_line_plot_layout.addWidget(self.display.filtered_cadence_line_plot)

        self.filtered_cadence_line_plot_widget = QtWidgets.QWidget(self.plots_tab)
        self.filtered_cadence_line_plot_widget.setLayout(self.filtered_cadence_line_plot_layout)
        self.filtered_cadence_line_plot_widget.setGeometry(QtCore.QRect(90, 10, 619, 403))
        # self.filtered_cadence_line_plot_widget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        ''' Unfiltered cadence line plot '''
        self.unfiltered_cadence_line_toolbar = NavigationToolbar(self.display.unfiltered_cadence_line_plot, self)
        self.unfiltered_cadence_line_plot_layout = QtWidgets.QVBoxLayout()
        self.unfiltered_cadence_line_plot_layout.addWidget(self.unfiltered_cadence_line_toolbar)
        self.unfiltered_cadence_line_plot_layout.addWidget(self.display.unfiltered_cadence_line_plot)

        self.unfiltered_cadence_line_plot_widget = QtWidgets.QWidget(self.plots_tab)
        self.unfiltered_cadence_line_plot_widget.setLayout(self.unfiltered_cadence_line_plot_layout)
        self.unfiltered_cadence_line_plot_widget.setGeometry(QtCore.QRect(90, 10, 619, 403))
        # self.unfiltered_cadence_line_plot_widget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        ''' Unfiltered cadence cadence plot '''
        self.unfiltered_cadence_scatter_plot = self.display.unfiltered_cadence_scatter_plot
        self.unfiltered_cadence_scatter_toolbar = NavigationToolbar(self.unfiltered_cadence_scatter_plot, self)
        self.unfiltered_cadence_scatter_plot_layout = QtWidgets.QVBoxLayout()
        self.unfiltered_cadence_scatter_plot_layout.addWidget(self.unfiltered_cadence_scatter_toolbar)
        self.unfiltered_cadence_scatter_plot_layout.addWidget(self.display.unfiltered_cadence_scatter_plot)

        self.unfiltered_cadence_scatter_plot_widget = QtWidgets.QWidget(self.plots_tab)
        self.unfiltered_cadence_scatter_plot_widget.setLayout(self.unfiltered_cadence_scatter_plot_layout)
        self.unfiltered_cadence_scatter_plot_widget.setGeometry(QtCore.QRect(90, 10, 619, 403))
        self.unfiltered_cadence_scatter_plot_widget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        ''' Add the plots to a plot stack '''
        self.plot_stack = QStackedLayout()
        self.plot_stack.addWidget(self.filtered_cadence_scatter_plot_widget)  # Index 0
        self.plot_stack.addWidget(self.filtered_cadence_line_plot_widget)  # Index 1
        self.plot_stack.addWidget(self.unfiltered_cadence_line_plot_widget)  # Index 2
        self.plot_stack.addWidget(self.unfiltered_cadence_scatter_plot_widget)  # Index 3
        '''
        
        self.frame_3 = QtWidgets.QFrame(self.plots_tab)
        self.frame_3.setGeometry(QtCore.QRect(90, 10, 619, 403))
        self.frame_3.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.frame_3.setAutoFillBackground(True)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        '''

    def set_plot_dropdown(self, text):
        self.plot = text
        print("{} selected".format(text))
        if text == "Filtered Cadence Scatter Plot":
            print("Setting {}".format(text))
            self.plot_stack.setCurrentIndex(0)


        elif text == "Filtered Cadence Line Plot":
            print("Setting {}".format(text))
            self.plot_stack.setCurrentIndex(1)

        elif text == "Unfiltered Cadence Line Plot":
            print("Setting {}".format(text))
            self.plot_stack.setCurrentIndex(2)

        elif text == "Unfiltered Cadence Scatter Plot":
            print("Setting {}".format(text))
            self.plot_stack.setCurrentIndex(3)

    def plot_dropdown_init(self):

        self.plot_dropdown = QtWidgets.QComboBox(self.plots_tab)
        self.plot_dropdown.setGeometry(QtCore.QRect(16, 466, 259, 41))
        self.plot_dropdown.setObjectName("plot_dropdown")

        self.plot_dropdown.addItem("Filtered Cadence Scatter Plot")
        self.plot_dropdown.addItem("Filtered Cadence Line Plot")
        self.plot_dropdown.addItem("Unfiltered Cadence Line Plot")
        self.plot_dropdown.addItem("Unfiltered Cadence Scatter Plot")

        self.plot_dropdown.activated[str].connect(self.set_plot_dropdown)

        self.pushButton_5 = QtWidgets.QPushButton(self.plots_tab)
        self.pushButton_5.setGeometry(QtCore.QRect(276, 464, 87, 45))
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_10 = QtWidgets.QLabel(self.plots_tab)
        self.label_10.setGeometry(QtCore.QRect(144, 424, 47, 13))
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.plots_tab)
        self.label_11.setGeometry(QtCore.QRect(20, 442, 169, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.p_output_lineEdit = QtWidgets.QLineEdit(self.plots_tab)
        self.p_output_lineEdit.setGeometry(QtCore.QRect(418, 464, 241, 43))
        self.p_output_lineEdit.setObjectName("p_output_lineEdit")
        self.label_12 = QtWidgets.QLabel(self.plots_tab)
        self.label_12.setGeometry(QtCore.QRect(416, 440, 169, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.pushButton_16 = QtWidgets.QPushButton(self.plots_tab)
        self.pushButton_16.setGeometry(QtCore.QRect(660, 462, 87, 45))
        self.pushButton_16.setObjectName("pushButton_16")
        self.tabWidget.addTab(self.plots_tab, "")
        self.MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self.MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 799, 21))
        self.menubar.setObjectName("menubar")
        self.MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(self.MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.height_label.setToolTip(_translate("MainWindow", "Enter the person\'s height in mm"))
        self.height_label.setText(_translate("MainWindow", "Enter height: (mm)"))
        self.calibrate_height_pushButton.setText(_translate("MainWindow", "Calibrate!"))
        self.weight_label_2.setText(_translate("MainWindow", "(Optional)"))
        self.weight_label_3.setText(_translate("MainWindow", "Enter weight: (Kg)"))
        self.weight_pushButton_2.setText(_translate("MainWindow", "Enter"))
        self.calibrate_frame_lineEdit_2.setText(_translate("MainWindow", "Frame path here"))
        self.calibrate_frame_pushButton_2.setText(_translate("MainWindow", "Enter"))
        self.calibrate_browsefile_pushButton.setText(_translate("MainWindow", "Browse"))
        self.calibrate_selectframe_label_2.setToolTip(
            _translate("MainWindow", "Select the frame path to calibrate the measurements"))
        self.calibrate_selectframe_label_2.setText(_translate("MainWindow", "Select frame:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.calibrate_tab), _translate("MainWindow", "Calibrate"))
        self.ratio_label.setText(_translate("MainWindow", "Pixel to millimetre ratio: N/A"))
        self.ratio_label.setToolTip(
            _translate("MainWindow", "The ratio between the chosen frame pixels and height in mm"))

        self.s_start_pushButton_3.setToolTip(_translate("MainWindow", "Start the operations!"))
        self.s_start_pushButton_3.setText(_translate("MainWindow", "Start"))
        self.s_vid_open_pushButton_6.setToolTip(_translate("MainWindow", "Open file location"))
        self.s_vid_open_pushButton_6.setText(_translate("MainWindow", "Open file"))
        self.s_play_pushButton_7.setToolTip(_translate("MainWindow", "Play/Pause the video"))
        self.s_play_pushButton_7.setText(_translate("MainWindow", "Play/Pause"))
        self.s_back_pushButton_8.setText(_translate("MainWindow", "Back"))
        self.s_forward_pushButton_9.setText(_translate("MainWindow", "Forward"))
        self.s_angle_checkBox.setToolTip(_translate("MainWindow", "Add the angle between \"\""))
        self.s_angle_checkBox.setText(_translate("MainWindow", "Angle"))
        self.s_lknee_angle_checkBox_2.setToolTip(_translate("MainWindow", "Add the angle between \"\""))
        self.s_lknee_angle_checkBox_2.setText(_translate("MainWindow", "Left knee angle"))
        self.s_rknee_angle_checkBox_3.setToolTip(_translate("MainWindow", "Add the angle between \"\""))
        self.s_rknee_angle_checkBox_3.setText(_translate("MainWindow", "Right knee angle"))
        self.s_legangle_checkBox_4.setToolTip(_translate("MainWindow", "Add the angle between \"\""))
        self.s_legangle_checkBox_4.setText(_translate("MainWindow", "Leg body angle"))
        self.s_progressBar.setToolTip(_translate("MainWindow", "<html><head/><body><p>statustooltip</p></body></html>"))
        self.s_saveoutput_checkBox_5.setToolTip(_translate("MainWindow", "Save the output to specified path"))
        self.s_saveoutput_checkBox_5.setText(_translate("MainWindow", "Save output:"))
        self.s_saveoutput_lineEdit_3.setText(_translate("MainWindow", ""))
        self.s_dp1_label_4.setToolTip(_translate("MainWindow", "Select the first point to measure distance between"))
        self.s_dp1_label_4.setText(_translate("MainWindow", "Distance point 1:"))
        self.s_dp2_label_5.setToolTip(_translate("MainWindow", "Select the second point to measure distance between"))
        self.s_dp2_label_5.setText(_translate("MainWindow", "Distance point 2:"))
        self.s_distance_checkBox_6.setToolTip(_translate("MainWindow", "Add the distance between two joints"))
        self.s_distance_checkBox_6.setText(_translate("MainWindow", "Distance:"))
        self.s_saveoutput_browse_pushButton.setText(_translate("MainWindow", "Browse"))
        self.s_measurements_label_6.setToolTip(_translate("MainWindow", "Select the measurements to find"))
        self.s_measurements_label_6.setText(_translate("MainWindow", "Select measurements:"))
        self.s_playback_label_7.setToolTip(_translate("MainWindow", "Select the measurements to find"))
        self.s_playback_label_7.setText(_translate("MainWindow", "Playback controls"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.saggital_tab), _translate("MainWindow", "Sagittal"))

        self.c_saveoutput_pushButton.setText(_translate("MainWindow", "Browse"))
        self.c_footangle_checkBox.setToolTip(_translate("MainWindow", "Add the angle between \"\""))
        self.c_footangle_checkBox.setText(_translate("MainWindow", "Foot angle"))
        self.c_output_lineEdit.setText(_translate("MainWindow", ""))
        self.progressBar_2.setToolTip(_translate("MainWindow", "<html><head/><body><p>statustooltip</p></body></html>"))
        self.c_start_pushButton.setToolTip(_translate("MainWindow", "Start the operations!"))
        self.c_start_pushButton.setText(_translate("MainWindow", "Start"))
        self.c_stepwidth_checkBox.setToolTip(_translate("MainWindow", "Add the angle between \"\""))
        self.c_stepwidth_checkBox.setText(_translate("MainWindow", "Step width"))
        self.c_saveoutput_checkBox.setToolTip(_translate("MainWindow", "Save the output to specified path"))
        self.c_saveoutput_checkBox.setText(_translate("MainWindow", "Save output:"))
        self.c_forward_pushButton.setText(_translate("MainWindow", "Forward"))
        self.c_openfile_pushButton.setToolTip(_translate("MainWindow", "Open file location"))
        self.c_openfile_pushButton.setText(_translate("MainWindow", "Open file"))
        self.c_back_pushButton.setText(_translate("MainWindow", "Back"))
        self.c_playback_label.setToolTip(_translate("MainWindow", "Select the measurements to find"))
        self.c_playback_label.setText(_translate("MainWindow", "Playback controls"))
        self.c_playback_pushButton.setToolTip(_translate("MainWindow", "Play/Pause the video"))
        self.c_playback_pushButton.setText(_translate("MainWindow", "Play/Pause"))
        self.c_measurements_label.setToolTip(_translate("MainWindow", "Select the measurements to find"))
        self.c_measurements_label.setText(_translate("MainWindow", "Select measurements:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.coronal_tab), _translate("MainWindow", "Coronal"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.metrics_tab), _translate("MainWindow", "Metrics"))

        self.pushButton_5.setText(_translate("MainWindow", "Enter"))
        self.label_11.setToolTip(_translate("MainWindow", "Select the plot using the drop down menu"))
        self.label_11.setText(_translate("MainWindow", "Select figure:"))
        self.p_output_lineEdit.setText(_translate("MainWindow", "Enter output path here"))
        self.label_12.setToolTip(_translate("MainWindow", "Save the output at the specified file location"))
        self.label_12.setText(_translate("MainWindow", "Save output:"))
        self.pushButton_16.setToolTip(_translate("MainWindow", "Save the figure"))
        self.pushButton_16.setText(_translate("MainWindow", "Enter"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.plots_tab), _translate("MainWindow", "Plots"))


class Worker3(QThread):
    """
       Worker thread for processing calibrating height
       """

    # finish_signal3 = pyqtSignal()

    def __init__(self, gui, parent=None):
        # super(External2, self).__init__()
        QThread.__init__(self, parent)
        self.gui = gui
        self.start()

    def run(self):
        # Remove any current images in output file
        points = self.gui.display.select_points()
        # self.finish_signal3.emit()
        self.gui.points = points
        # self.finish_signal3.emit()
        self.gui.image_label.setPixmap(QtGui.QPixmap("temp_calibration_image.png"))
        distance = bf.get_y_distance(points[0], points[1])
        self.gui.pixel_distance = distance
        print("Pixel distance:", distance)
        self.gui.pixels_to_mm()
        self.quit()


TIME_LIMIT = 2400000000000000


class CoronalExternal(QThread):
    """
    run the progress bar in an external thread class
    """
    mySignal = pyqtSignal(int)

    def __init__(self, gui):
        super(CoronalExternal, self).__init__()
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
        self.start()

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
                print(
                    "Saving to video . . . Progress ({}, {}) / {} frames. {}% complete. {} Frames remaining . . .".format(
                        self.frame, self.gui.display.frame_number, self.gui.frame_count, self.progress,
                        self.gui.frame_count - self.frame))
                # Increase the progress bar and set to new frame
                self.frame = self.gui.display.frame_number
                self.progress += add
                # self.gui.onCountChanged(self.progress)
                self.mySignal.emit(self.progress)
                if self.gui.frame_count - self.frame < 1:
                    print("Process complete . . . Please wait.")
                    self.quit()
        self.quit()
        print("Timer finished")


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
        self.start()

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
                print(
                    "Saving to video . . . Progress ({}, {}) / {} frames. {}% complete. {} Frames remaining . . .".format(
                        self.frame, self.gui.display.frame_number, self.gui.frame_count, self.progress,
                        self.gui.frame_count - self.frame))
                # Increase the progress bar and set to new frame
                self.frame = self.gui.display.frame_number
                self.progress += add
                # self.gui.onCountChanged(self.progress)
                self.mySignal.emit(self.progress)
                if self.gui.frame_count - self.frame < 1:
                    print("Process complete . . . Please wait.")
                    self.quit()
        self.quit()
        print("Timer finished")


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
        self.start()

    def run(self):
        # Remove existing files
        files = glob.glob("{}\\*.png".format("output_images"))
        for f in files:
            os.remove(f)
        files = glob.glob("{}\\*.png".format("display_images"))
        for f in files:
            os.remove(f)
        start = 0

        if self.gui.s_distance_checkBox_6 == Qt.Checked:
            start = 1
            # self.gui.display.horizontal_foot_angle_overlay()
            self.gui.display.distance_overlay()

        if self.gui.s_angle_checkBox == Qt.Checked:
            start = 1
            self.gui.display.angle_overlay()

        if self.gui.s_legangle_checkBox_4 == Qt.Checked:
            start = 1
            self.gui.display.leg_body_angle_overlay()

        if self.gui.s_lknee_angle_checkBox_2 == Qt.Checked:
            start = 1
            self.gui.display.left_knee_angle_overlay()

        if self.gui.s_rknee_angle_checkBox_3 == Qt.Checked:
            start = 1
            self.gui.display.right_knee_angle_overlay()

        if start == 0:
            # If no options selected send a signal
            self.start_signal.emit()

        else:
            # self.gui.display.display_step_number_overlay()
            if self.gui.s_saveoutput_checkBox_5 == Qt.Checked:
                for frame in self.gui.display.frame_list:
                    self.gui.s_progressBar.setStatusTip(
                        "Operations complete. Saving all frames... This may take awhile..")
                    bf.save_frame(frame)
            else:
                print("Option to save frames not selected, moving on. . . ")
            # noinspection PyBroadException
            try:
                if self.gui.s_saveoutput_checkBox_5 == Qt.Checked:
                    self.gui.s_progressBar.setStatusTip("Operations complete. Saving video... This may take awhile..")
                    bf.save_video(self.gui.s_video_output_path)
                else:
                    print("Option to save video not selected, moving on. . . ")
                try:
                    # self.output_images = os.listdir('output_images/')
                    # self.gui.s_vid.stop()
                    # THIS METHOD IS VERY SLOW !! !!! !!
                    '''
                    count = 1
                    self.gui.max_preview_count = len(self.output_images)
                    for path in self.output_images:
                        bf.resize_image('output_images/{}'.format(self.output_images[count]), 409, 311, count)
                        count += 1

                    img_path = 'display_images/1.png'
                    pixmap = QPixmap(img_path)
                    self.gui.s_vid_frame.setPixmap(pixmap)
                    '''
                    '''
                    self.s_vid = QMovie("Output.mp4")
                    self.s_vid_frame.setMovie(self.s_vid)
                    self.s_vid.start()
                    '''
                    print("Trying to add video preview")
                    print("Trying toset vid out")


                except Exception as e:
                    print("failed to set new movie")
                    raise e
                self.finish_signal.emit()

            except Exception as e:
                print("quiting...")
                raise e
            # self.process_complete_messagebox()
        self.quit()

def handler(msg_type, msg_log_context, msg_string):
    pass
QtCore.qInstallMessageHandler(handler)

class CoronalWorker(QThread):
    """
    Worker thread for processing measurements in the sagittal plane
    """
    start_signal = pyqtSignal()
    finish_signal = pyqtSignal()

    def __init__(self, gui, parent=None):
        # super(External2, self).__init__()
        QThread.__init__(self, parent)
        self.gui = gui
        self.start()

    def run(self):
        # Remove existing files
        files = glob.glob("{}\\*.png".format("output_coronal_images"))
        for f in files:
            os.remove(f)
        files = glob.glob("{}\\*.png".format("display_images"))
        for f in files:
            os.remove(f)
        start = 0

        if self.gui.c_stepwidth_checkBox == Qt.Checked:
            print("pass")
            start = 1
            self.gui.display.display_step_width()

        if self.gui.c_footangle_checkBox == Qt.Checked:
            start = 1
            self.gui.display.foot_angle_overlay()

        if start == 0:
            print("no pass")
            # If no options selected send a signal
            self.start_signal.emit()

        else:

            if self.gui.c_saveoutput_checkBox == Qt.Checked:
                for frame in self.gui.display.coronal_frame_list:
                    self.gui.progressBar_2.setStatusTip(
                        "Operations complete. Saving all frames... This may take awhile..")
                    bf.coronal_save_frame(frame)
            else:
                print("Option to save frames not selected, moving on. . . ")
            # noinspection PyBroadException
            try:
                if self.gui.c_saveoutput_checkBox == Qt.Checked:
                    self.gui.progressBar_2.setStatusTip("Operations complete. Saving video... This may take awhile..")
                    bf.coronal_save_video(self.gui.c_video_output_path)
                else:
                    print("Option to save video not selected, moving on. . . ")

                self.finish_signal.emit()

            except Exception as e:
                print("quiting...")
                raise e
            # self.process_complete_messagebox()
        self.quit()


if __name__ == "__main__":
    import sys

    display = 1
    app = QtWidgets.QApplication(sys.argv)
