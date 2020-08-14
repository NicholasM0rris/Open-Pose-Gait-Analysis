import os
from os import startfile
import sys
import glob
import support.base_functions as bf
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets



class GUI(QMainWindow):

    def __init__(self, display):
        super(GUI, self).__init__()
        self.num_operations = 0
        self.calc = None
        self.output_movie = ""
        self.display = display
        # self.display.gui = self
        # self.frame_count = self.display.frame_count
        # self.app = QApplication([]) (not needed?)
        self.height = 0
        self.pixel_distance = 0
        self.points = []

        self.MainWindow = QtWidgets.QMainWindow()
        # ui = GUI(display)
        self.setupUi(self.MainWindow)
        self.MainWindow.setWindowTitle("Early development user interface A204 V2.31")
        QApplication.setStyle(QStyleFactory.create("Fusion"))
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

    def pixels_to_mm(self):
        self.pixel_ratio = self.height / self.pixel_distance
        print("One pixel is equivalent to {} mms".format(self.pixel_ratio))

    def calibrateheightbuttonclick(self):
        try:
            self.height = int(self.height_lineEdit.text())
            self.calibrate_thread = Worker3(self)
            self.calibrate_thread.start()
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



        #self.frame = QtWidgets.QFrame(self.calibrate_tab)
        #self.frame.setGeometry(QtCore.QRect(128, 60, 543, 353))
        #self.frame.setAutoFillBackground(True)
        #self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        #self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        #self.frame.setObjectName("frame")

        self.calibrate_frame_lineEdit_2 = QtWidgets.QLineEdit(self.calibrate_tab)
        self.calibrate_frame_lineEdit_2.setGeometry(QtCore.QRect(130, 22, 113, 20))
        self.calibrate_frame_lineEdit_2.setObjectName("calibrate_frame_lineEdit_2")
        self.calibrate_frame_pushButton_2 = QtWidgets.QPushButton(self.calibrate_tab)
        self.calibrate_frame_pushButton_2.setGeometry(QtCore.QRect(244, 22, 75, 21))
        self.calibrate_frame_pushButton_2.setObjectName("calibrate_frame_pushButton_2")
        self.calibrate_selectframe_label_2 = QtWidgets.QLabel(self.calibrate_tab)
        self.calibrate_selectframe_label_2.setGeometry(QtCore.QRect(12, 14, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.calibrate_selectframe_label_2.setFont(font)
        self.calibrate_selectframe_label_2.setObjectName("calibrate_selectframe_label_2")
        self.tabWidget.addTab(self.calibrate_tab, "")

    def init_sagittal_tab(self):
        self.saggital_tab = QtWidgets.QWidget()
        self.saggital_tab.setObjectName("saggital_tab")

        self.init_s_vid_frame()
        self.init_s_start_button()
        self.init_s_video()
        self.s_init_checkboxes()
        self.s_init_progressbar()
        self.s_init_save_output()
        self.s_init_dropdownbox()



    def init_s_vid_frame(self):
        self.s_vid_frame = QtWidgets.QFrame(self.saggital_tab)
        self.s_vid_frame.setGeometry(QtCore.QRect(178, 30, 409, 311))
        self.s_vid_frame.setAutoFillBackground(True)
        self.s_vid_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.s_vid_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.s_vid_frame.setObjectName("s_vid_frame")

    def init_s_start_button(self):
        self.s_start_pushButton_3 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_start_pushButton_3.setGeometry(QtCore.QRect(680, 478, 89, 41))
        self.s_start_pushButton_3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_start_pushButton_3.setObjectName("s_start_pushButton_3")

    def init_s_video(self):
        self.s_vid_open_pushButton_6 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_vid_open_pushButton_6.setGeometry(QtCore.QRect(620, 202, 131, 43))
        self.s_vid_open_pushButton_6.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_vid_open_pushButton_6.setObjectName("s_vid_open_pushButton_6")

        self.s_play_pushButton_7 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_play_pushButton_7.setGeometry(QtCore.QRect(620, 62, 131, 43))
        self.s_play_pushButton_7.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_play_pushButton_7.setObjectName("s_play_pushButton_7")

        self.s_back_pushButton_8 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_back_pushButton_8.setGeometry(QtCore.QRect(620, 132, 65, 43))
        self.s_back_pushButton_8.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_back_pushButton_8.setObjectName("s_back_pushButton_8")

        self.s_forward_pushButton_9 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_forward_pushButton_9.setGeometry(QtCore.QRect(684, 132, 67, 43))
        self.s_forward_pushButton_9.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_forward_pushButton_9.setObjectName("s_forward_pushButton_9")



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
            self.lknee_angle_checkBox_2 = Qt.Checked
            print('left knee Angle Checked')
        else:
            self.lknee_angle_checkBox_2 = Qt.Unchecked
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
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_saveoutput_checkBox_5.setFont(font)
        self.s_saveoutput_checkBox_5.setObjectName("s_saveoutput_checkBox_5")
        self.s_saveoutput_lineEdit_3 = QtWidgets.QLineEdit(self.saggital_tab)
        self.s_saveoutput_lineEdit_3.setGeometry(QtCore.QRect(476, 436, 233, 29))
        self.s_saveoutput_lineEdit_3.setObjectName("s_saveoutput_lineEdit_3")
        self.s_saveoutput_pushButton_10 = QtWidgets.QPushButton(self.saggital_tab)
        self.s_saveoutput_pushButton_10.setGeometry(QtCore.QRect(712, 436, 77, 29))
        self.s_saveoutput_pushButton_10.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.s_saveoutput_pushButton_10.setObjectName("s_saveoutput_pushButton_10")

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

    def coronal_tab_init(self):
        self.coronal_tab = QtWidgets.QWidget()
        self.coronal_tab.setObjectName("coronal_tab")
        self.pushButton_11 = QtWidgets.QPushButton(self.coronal_tab)
        self.pushButton_11.setGeometry(QtCore.QRect(712, 436, 77, 29))
        self.pushButton_11.setObjectName("pushButton_11")
        self.checkBox_8 = QtWidgets.QCheckBox(self.coronal_tab)
        self.checkBox_8.setGeometry(QtCore.QRect(22, 418, 167, 41))
        self.checkBox_8.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.checkBox_8.setFont(font)
        self.checkBox_8.setIconSize(QtCore.QSize(16, 16))
        self.checkBox_8.setObjectName("checkBox_8")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.coronal_tab)
        self.lineEdit_4.setGeometry(QtCore.QRect(476, 436, 233, 29))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.progressBar_2 = QtWidgets.QProgressBar(self.coronal_tab)
        self.progressBar_2.setGeometry(QtCore.QRect(14, 480, 661, 37))
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setObjectName("progressBar_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.coronal_tab)
        self.pushButton_4.setGeometry(QtCore.QRect(680, 478, 89, 41))
        self.pushButton_4.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_4.setObjectName("pushButton_4")
        self.checkBox_11 = QtWidgets.QCheckBox(self.coronal_tab)
        self.checkBox_11.setGeometry(QtCore.QRect(22, 366, 167, 53))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.checkBox_11.setFont(font)
        self.checkBox_11.setToolTipDuration(-1)
        self.checkBox_11.setStatusTip("")
        self.checkBox_11.setWhatsThis("")
        self.checkBox_11.setObjectName("checkBox_11")
        self.frame_2 = QtWidgets.QFrame(self.coronal_tab)
        self.frame_2.setGeometry(QtCore.QRect(178, 30, 409, 311))
        self.frame_2.setAutoFillBackground(True)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.checkBox_12 = QtWidgets.QCheckBox(self.coronal_tab)
        self.checkBox_12.setGeometry(QtCore.QRect(476, 398, 149, 35))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_12.setFont(font)
        self.checkBox_12.setObjectName("checkBox_12")
        self.pushButton_12 = QtWidgets.QPushButton(self.coronal_tab)
        self.pushButton_12.setGeometry(QtCore.QRect(684, 132, 67, 43))
        self.pushButton_12.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(self.coronal_tab)
        self.pushButton_13.setGeometry(QtCore.QRect(620, 202, 131, 43))
        self.pushButton_13.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(self.coronal_tab)
        self.pushButton_14.setGeometry(QtCore.QRect(620, 132, 65, 43))
        self.pushButton_14.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_14.setObjectName("pushButton_14")
        self.label_8 = QtWidgets.QLabel(self.coronal_tab)
        self.label_8.setGeometry(QtCore.QRect(616, 32, 169, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.pushButton_15 = QtWidgets.QPushButton(self.coronal_tab)
        self.pushButton_15.setGeometry(QtCore.QRect(620, 62, 131, 43))
        self.pushButton_15.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_15.setObjectName("pushButton_15")
        self.label_9 = QtWidgets.QLabel(self.coronal_tab)
        self.label_9.setGeometry(QtCore.QRect(16, 350, 169, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.tabWidget.addTab(self.coronal_tab, "")

        self.metrics_tab = QtWidgets.QWidget()
        self.metrics_tab.setObjectName("metrics_tab")
        self.tabWidget.addTab(self.metrics_tab, "")
        self.plots_tab = QtWidgets.QWidget()
        self.plots_tab.setObjectName("plots_tab")
        self.frame_3 = QtWidgets.QFrame(self.plots_tab)
        self.frame_3.setGeometry(QtCore.QRect(90, 10, 619, 403))
        self.frame_3.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.frame_3.setAutoFillBackground(True)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.comboBox_2 = QtWidgets.QComboBox(self.plots_tab)
        self.comboBox_2.setGeometry(QtCore.QRect(16, 466, 259, 41))
        self.comboBox_2.setObjectName("comboBox_2")
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
        self.lineEdit_5 = QtWidgets.QLineEdit(self.plots_tab)
        self.lineEdit_5.setGeometry(QtCore.QRect(418, 464, 241, 43))
        self.lineEdit_5.setObjectName("lineEdit_5")
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
        self.calibrate_selectframe_label_2.setToolTip(
            _translate("MainWindow", "Select the frame path to calibrate the measurements"))
        self.calibrate_selectframe_label_2.setText(_translate("MainWindow", "Select frame:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.calibrate_tab), _translate("MainWindow", "Calibrate"))

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
        self.s_saveoutput_lineEdit_3.setText(_translate("MainWindow", "Enter output path here"))
        self.s_dp1_label_4.setToolTip(_translate("MainWindow", "Select the first point to measure distance between"))
        self.s_dp1_label_4.setText(_translate("MainWindow", "Distance point 1:"))
        self.s_dp2_label_5.setToolTip(_translate("MainWindow", "Select the second point to measure distance between"))
        self.s_dp2_label_5.setText(_translate("MainWindow", "Distance point 2:"))
        self.s_distance_checkBox_6.setToolTip(_translate("MainWindow", "Add the distance between two joints"))
        self.s_distance_checkBox_6.setText(_translate("MainWindow", "Distance:"))
        self.s_saveoutput_pushButton_10.setText(_translate("MainWindow", "Enter"))
        self.s_measurements_label_6.setToolTip(_translate("MainWindow", "Select the measurements to find"))
        self.s_measurements_label_6.setText(_translate("MainWindow", "Select measurements:"))
        self.s_playback_label_7.setToolTip(_translate("MainWindow", "Select the measurements to find"))
        self.s_playback_label_7.setText(_translate("MainWindow", "Playback controls"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.saggital_tab), _translate("MainWindow", "Sagittal"))

        self.pushButton_11.setText(_translate("MainWindow", "Enter"))
        self.checkBox_8.setToolTip(_translate("MainWindow", "Add the angle between \"\""))
        self.checkBox_8.setText(_translate("MainWindow", "Foot angle"))
        self.lineEdit_4.setText(_translate("MainWindow", "Enter output path here"))
        self.progressBar_2.setToolTip(_translate("MainWindow", "<html><head/><body><p>statustooltip</p></body></html>"))
        self.pushButton_4.setToolTip(_translate("MainWindow", "Start the operations!"))
        self.pushButton_4.setText(_translate("MainWindow", "Start"))
        self.checkBox_11.setToolTip(_translate("MainWindow", "Add the angle between \"\""))
        self.checkBox_11.setText(_translate("MainWindow", "Step width"))
        self.checkBox_12.setToolTip(_translate("MainWindow", "Save the output to specified path"))
        self.checkBox_12.setText(_translate("MainWindow", "Save output:"))
        self.pushButton_12.setText(_translate("MainWindow", "Forward"))
        self.pushButton_13.setToolTip(_translate("MainWindow", "Open file location"))
        self.pushButton_13.setText(_translate("MainWindow", "Open file"))
        self.pushButton_14.setText(_translate("MainWindow", "Back"))
        self.label_8.setToolTip(_translate("MainWindow", "Select the measurements to find"))
        self.label_8.setText(_translate("MainWindow", "Playback controls"))
        self.pushButton_15.setToolTip(_translate("MainWindow", "Play/Pause the video"))
        self.pushButton_15.setText(_translate("MainWindow", "Play/Pause"))
        self.label_9.setToolTip(_translate("MainWindow", "Select the measurements to find"))
        self.label_9.setText(_translate("MainWindow", "Select measurements:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.coronal_tab), _translate("MainWindow", "Coronal"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.metrics_tab), _translate("MainWindow", "Metrics"))

        self.pushButton_5.setText(_translate("MainWindow", "Enter"))
        self.label_11.setToolTip(_translate("MainWindow", "Select the plot using the drop down menu"))
        self.label_11.setText(_translate("MainWindow", "Select figure:"))
        self.lineEdit_5.setText(_translate("MainWindow", "Enter output path here"))
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
        print("Timer finished")


if __name__ == "__main__":
    import sys

    display = 1
    app = QtWidgets.QApplication(sys.argv)