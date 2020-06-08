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

        self.r_mtx, self.r_dist, self.r_tvecs, self.r_rvecs = self.right_calibrate()
        self.l_mtx, self.l_dist, self.l_tvecs, self.l_rvecs = self.left_calibrate()

        self.camera_model = self.stereo_calibrate()

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

        camera_model = dict([('l_mtx', l_mtx), ('r_mtx', r_mtx), ('dist1', l_dist),
                             ('dist2', r_dist), ('rvecs1', self.l_rvecs),
                             ('rvecs2', self.r_rvecs), ('R', R), ('T', T),
                             ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        print("Success")
        return camera_model


def main(argv=None):
    calibrate = Calibration()


if __name__ == '__main__':
    main(sys.argv[1:])
