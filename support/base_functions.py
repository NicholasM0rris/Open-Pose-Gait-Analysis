import sys
import cv2
from datetime import datetime
import numpy as np
import glob
import os
import argparse
from main import args

def save_frame(frame):
    """
    Save a frame to output_images
    :param frame:
    :return:
    """
    time_stamp = datetime.now()
    filename = "{}.png".format(time_stamp.strftime("%Y-%m-%d_%H-%M-%S-%f"))
    path = "output_images\\{}".format(filename)
    cv2.imwrite(path, frame)


def save_frame2(frame):
    """
    Save a frame to output_images
    :param frame:
    :return:
    """
    time_stamp = datetime.now()
    filename = "{}.png".format(time_stamp.strftime("%Y-%m-%d_%H-%M-%S-%f"))
    path = "output_coronal_images\\{}".format(filename)
    cv2.imwrite(path, frame)


def add_line_between_points(frame, points):
    """
    Adds a line overlay between two points and puts pixel distance text
    :param frame:
    :param points:
    :return:
    """
    point1 = list(map(int, points[0]))
    point2 = list(map(int, points[1]))
    cv2.line(frame, tuple(point1), tuple(point2), (0, 255, 0), thickness=3, lineType=8)
    org = tuple(point1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    color = (0, 0, 255)
    thickness = 2
    print("pt1,pt2", point1, point2)
    frame = cv2.putText(frame, 'Distance: {}'.format(get_distance(point1, point2)), org, font,
                        fontscale, color, thickness, cv2.LINE_AA)
    return frame


def add_points_to_image(frame, keypoints):
    """
    Overlay points to an image
    :param frame: Image/frame for points to be overlayed (from extract_frame) in red
    :param keypoints: list of keypoint coordinates to overlay
    :return: writes frame to output_images
    """
    for keypoint in keypoints:
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 10, (0, 0, 255), -1)
    return frame


def get_distance(point1, point2):
    """

    :param point1: list of coordinates
    :param point2:
    :return:
    """
    print("ptt2", point1, point2)
    dist = np.linalg.norm(np.asarray(point1) - np.asarray(point2))
    print("Distance between two points: ", dist)
    return dist


def distance_overlay(display, point1, point2):
    """
    Creates overlay for distance
    :param display:
    :param point1:
    :param point2:
    :return:
    """
    # Remove any current images in output file
    files = glob.glob("{}\\*.png".format("output_images"))
    for f in files:
        os.remove(f)
    # Add overlay
    for idx, path in enumerate(display.data.input_files):
        frame = cv2.imread(path)
        frame = add_points_to_image(frame, [display.fp(point1, idx), display.fp(point2, idx)])
        frame = add_line_between_points(frame, [display.fp(point1, idx), display.fp(point2, idx)])
        save_frame(frame)
    save_video()


def save_video():
    """
    Saves a video of processed image output to processed_video directory
    :return:
    """
    images = []
    for filename in glob.glob("{}\\*.png".format("output_images")):
        images.append(filename)
    try:
        frame = cv2.imread(images[0])
    except IndexError:
        print("Index error: No images in output folder")
        sys.exit("Index error: No images in output folder")
    height, width, layers = frame.shape
    if args['fps']:
        video = cv2.VideoWriter("Output.avi", 0, args['fps'], (width, height))
    else:
        video = cv2.VideoWriter("Output.avi", 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


def save_video2():
    """
    Saves a video of processed image output to coronal processed_video directory
    :return:
    """
    images = []
    for filename in glob.glob("{}\\*.png".format("output_coronal_images")):
        images.append(filename)
    try:
        frame = cv2.imread(images[0])
    except IndexError:
        print("Index error2: No images in output folder")
        sys.exit("Index error: No images in output folder")
    height, width, layers = frame.shape
    # video = cv2.VideoWriter("{}/Coronal_Output.avi".format("processed_video"), 0, 1, (width, height))
    if args['fps']:
        video = cv2.VideoWriter("Coronal_Output.avi", 0, args['fps'], (width, height))
    else:
        video = cv2.VideoWriter("Coronal_Output.avi", 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


def get_mag(pt1):
    return (pt1[0] ** 2 + pt1[1] ** 2) ** 0.5


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
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', required=False, help='Add image directory')
    ap.add_argument('-v', '--video', required=False, help='Add Video')
    ap.add_argument('-d', '--data', required=False, help='Add data directory')
    ap.add_argument('-cd', '--cdata', required=False, help='Add coronal data directory')
    ap.add_argument('-ci', '--cimages', required=False, help='Add coronal image directory')
    ap.add_argument('-height', '--height', required=False, type=int,
                    help='Add height of the person in centimetres (cm)')
    ap.add_argument('-fps', '--fps', required=False, type=int, help='FPS to save video output')
    ap.add_argument('-vl', '-video_length', required=False, type=float, help='Add the video length in seconds')

    args = vars(ap.parse_args())


if __name__ == '__main__':
    main(sys.argv[1:])
