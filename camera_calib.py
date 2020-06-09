import numpy as np
import cv2
import glob
import sys


class Calibration:
    def __init__(self):
        # Termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.left_objp = np.zeros((9 * 7, 3), np.float32)
        self.left_objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)
        self.right_objp = np.zeros((9 * 7, 3), np.float32)
        self.right_objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)
        self.left_objpoints = []  # 3d point in real world space
        self.left_imgpoints = []  # 2d points in image plane.
        self.right_objpoints = []  # 3d point in real world space
        self.right_imgpoints = []  # 2d points in image plane.
        self.left_images = glob.glob('support\\left_checkerboard\\*.jpg')
        self.right_images = glob.glob('support\\right_checkerboard\\*.jpg')
        # Distance between the camera (mm)
        self.base = 1800

        self.r_mtx, self.r_dist, self.r_tvecs, self.r_rvecs = self.right_calibrate()
        self.l_mtx, self.l_dist, self.l_tvecs, self.l_rvecs = self.left_calibrate()

        self.camera_model = self.stereo_calibrate()
        self.depth_map()

        print('Camera model', self.camera_model)

    def left_calibrate(self):

        for fname in self.left_images:
            left_img = cv2.imread(fname)
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            # Change 7 and 9 to match checkerboard inside squares
            left_ret, corners = cv2.findChessboardCorners(left_gray, (7, 9), None)

            # If found, add object points, image points (after refining them)
            if left_ret:
                self.left_objpoints.append(self.left_objp)

                corners2 = cv2.cornerSubPix(left_gray, corners, (11, 11), (-1, -1), self.criteria)
                self.left_imgpoints.append(corners2)

                # Draw and display the corners
                left_img = cv2.drawChessboardCorners(left_img, (7, 9), corners2, left_ret)
                cv2.imshow('img', left_img)
                cv2.imwrite('support\\calib_results\\calib_board.png', left_img)
                cv2.waitKey(500)
            else:
                print("Not a success")
        left_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.left_objpoints, self.left_imgpoints,
                                                                left_gray.shape[::-1], None, None)

        img = cv2.imread('support\\left_checkerboard\\image.jpg')
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imwrite('support\\calib_results\\calibresult.png', dst)

        total_error = 0
        for i in range(len(self.left_objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.left_objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.left_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print(error)
        print("Left total error: ", total_error / len(self.left_objpoints))

        return mtx, dist, rvecs, tvecs

    def right_calibrate(self):

        for fname in self.right_images:
            right_img = cv2.imread(fname)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            right_ret, corners = cv2.findChessboardCorners(right_gray, (7, 9), None)

            # If found, add object points, image points (after refining them)
            if right_ret:
                self.right_objpoints.append(self.right_objp)

                corners2 = cv2.cornerSubPix(right_gray, corners, (11, 11), (-1, -1), self.criteria)
                self.right_imgpoints.append(corners2)

                # Draw and display the corners
                right_img = cv2.drawChessboardCorners(right_img, (7, 9), corners2, right_ret)
                cv2.imshow('img', right_img)
                cv2.waitKey(500)
            else:
                print("Not a success")
        right_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.right_objpoints, self.right_imgpoints,
                                                                 right_gray.shape[::-1], None, None)
        self.dimensions = right_gray.shape[::-1]

        img = cv2.imread('support\\right_checkerboard\\image.jpg')
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imwrite('support\\calib_results\\calibresult.png', dst)

        total_error = 0
        for i in range(len(self.right_objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.right_objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.right_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print(error)
        print("Right total error: ", total_error / len(self.right_objpoints))

        return mtx, dist, rvecs, tvecs

    def stereo_calibrate(self):
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, l_mtx, l_dist, r_mtx, r_dist, R, T, E, F = cv2.stereoCalibrate(
            self.left_objpoints, self.left_imgpoints,
            self.right_imgpoints, self.l_mtx, self.l_dist, self.r_mtx,
            self.r_dist, self.dimensions,
            criteria=stereocalib_criteria)

        (l_rect, r_rect, l_proj, r_proj,
         dispartityToDepthMap, l_ROI, r_ROI) = cv2.stereoRectify(
            l_mtx, l_dist,
            r_mtx, r_dist,
            self.dimensions, R, T,
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY)

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            l_mtx, l_dist, l_rect,
            l_proj, self.dimensions, cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            r_mtx, r_dist, r_rect,
            r_proj, self.dimensions, cv2.CV_32FC1)

        # Save calibration to file
        np.savez_compressed("support\\calib_results\\calibration.npz", imageSize=self.dimensions,
                            leftMapX=leftMapX, leftMapY=leftMapY, leftROI=l_ROI,
                            rightMapX=rightMapX, rightMapY=rightMapY, rightROI=r_ROI)

        camera_model = dict([('l_mtx', l_mtx), ('r_mtx', r_mtx), ('dist1', l_dist),
                             ('dist2', r_dist), ('rvecs1', self.l_rvecs),
                             ('rvecs2', self.r_rvecs), ('R', R), ('T', T),
                             ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        print("Success")
        return camera_model

    def depth_map(self):
        """
        Create a depth map of the stereo cameras

            The camera matrix is of the following form:

            f_x  s    c_x
            0    f_y  c_y
            0    0    1
            where f_x is the camera focal length in the x axis in pixels

        f_y is the camera focal length in the y axis in pixels

        s is a skew parameter (normally not used)

        c_x is the optical center in x

        c_y is the optical center in y

        disparity=x−x′=Bf/Z
        x and x′ are the distance between points in image plane corresponding
        to the scene point 3D and their camera center. B is the distance between
        two cameras (which we know) and f is the focal length of camera (already known)
        """

        calibration = np.load("support\\calib_results\\calibration.npz")
        imageSize = tuple(calibration["imageSize"])
        leftMapX = calibration["leftMapX"]
        leftMapY = calibration["leftMapY"]
        leftROI = tuple(calibration["leftROI"])
        rightMapX = calibration["rightMapX"]
        rightMapY = calibration["rightMapY"]
        rightROI = tuple(calibration["rightROI"])

        stereoMatcher = cv2.StereoBM_create()

        left_frame = cv2.imread("support\\left_checkerboard\\image.jpg")
        cv2.imshow("left", left_frame)
        cv2.waitKey(1000)
        right_frame = cv2.imread("support\\right_checkerboard\\image.jpg")
        fixedLeft = cv2.remap(left_frame, leftMapX, leftMapY, cv2.INTER_LINEAR)
        fixedRight = cv2.remap(right_frame, rightMapX, rightMapY, cv2.INTER_LINEAR)

        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        depth = stereoMatcher.compute(grayLeft, grayRight)

        real_depth = (self.base * self.camera_model["l_mtx"][1][1]) / (depth[0][0] + 1)
        print("REAL DEPTH", real_depth)
        # Set parameters from albert's settings
        stereoMatcher.setMinDisparity(4)
        stereoMatcher.setNumDisparities(128)
        stereoMatcher.setBlockSize(21)
        stereoMatcher.setSpeckleRange(16)
        stereoMatcher.setSpeckleWindowSize(45)

        DEPTH_VISUALIZATION_SCALE = 2048
        print("DEPTH", type(depth))
        print("length", np.shape(depth))
        np.savetxt('support\\calib_results\\depth.txt', depth, fmt='%d')
        cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
        cv2.imwrite("support\\calib_results\\depth.png", depth)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def print_calibration(self):
        data = np.load('support\\calib_results\\calibration.npz')
        lst = data.files
        for item in lst:
            print(item)
            print(data[item])


def main(argv=None):
    calibrate = Calibration()
    # calibrate.print_calibration()


if __name__ == '__main__':
    main(sys.argv[1:])
