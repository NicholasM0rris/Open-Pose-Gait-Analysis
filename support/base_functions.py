import sys
import cv2
from datetime import datetime
import numpy as np
import glob
import re
import os
import argparse
from PIL import Image, ImageFilter
from main import args


def anonymise_images(frames, nose_points, right_ear_points, left_ear_points):
    """
    Add a Gaussian blur to conceal face for privacy reasons
    :param frames: The list of frames to process (Sagittal plane)
    :param nose: The key point to center the blur
    :return: List of path names for the blurred images
    """

    padx = 100
    pady = 80
    nose_x = 0
    nose_y = 0
    for idx, path in enumerate(frames):
        # print(len(frames))
        # print(frames)

        frame = Image.open(path)
        if nose_points[idx][0] != 0 and nose_points[idx][1] != 0:
            nose_x = nose_points[idx][0]
            nose_y = nose_points[idx][1]
            # print(nose_x, nose_y)

        elif left_ear_points[idx][0] != 0 and left_ear_points[idx][1] != 0:
            nose_x = left_ear_points[idx][0]
            nose_y = left_ear_points[idx][1]
        elif right_ear_points[idx][0] != 0 and right_ear_points[idx][1] != 0:
            nose_x = right_ear_points[idx][0]
            nose_y = right_ear_points[idx][1]
        else:
            # No point available - check previous point
            try:
                if nose_points[idx - 1][0] != 0 and nose_points[idx - 1][1] != 0:
                    nose_x = nose_points[idx - 1][0]
                    nose_y = nose_points[idx - 1][1]
                    # print(nose_x, nose_y)

                elif left_ear_points[idx - 1][0] != 0 and left_ear_points[idx - 1][1] != 0:
                    nose_x = left_ear_points[idx - 1][0]
                    nose_y = left_ear_points[idx][1]
                elif right_ear_points[idx - 1][0] != 0 and right_ear_points[idx - 1][1] != 0:
                    nose_x = right_ear_points[idx - 1][0]
                    nose_y = right_ear_points[idx - 1][1]
            except IndexError:
                nose_x = 0
                nose_y = 0

        point1 = nose_x - padx
        point2 = nose_y - pady
        if point1 < 0:
            point1 = 0
        if point2 < 0:
            point2 = 0

        nose = (int(point1), int(point2), int(nose_x + padx), int(nose_y + pady))
        cropped_frame = frame.crop(nose)
        blurred_frame = cropped_frame.filter(ImageFilter.GaussianBlur(radius=20))
        # print(nose)
        # sys.exit()
        frame.paste(blurred_frame, nose)

        outpath = "{}\\{}.png".format("blurred_images", idx + 1)
        print(outpath)
        print(idx, "idx")
        print("path", path)
        frame.save(outpath)
        if idx == 350:
            print(frames)
            print(len(frames))

    input_files = []
    print("Finished FOR loop")
    for filename in glob.glob("{}\\*.png".format("blurred_images")):
        input_files.append(filename)
    # print(input_files)
    # Stupid python input_files.sort(key=lambda x: int(float(os.path.basename(x).split('.')[0][1:])))
    print("Start sorting")
    input_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print("Finished blurring")
    return input_files


def anonymise_coronal_images(frames, nose_points, right_ear_points, left_ear_points):
    """
    Add a Gaussian blur to conceal face for privacy reasons
    :param frames: The list of frames to process
    :param nose: The key point to center the blur
    :return: List of path names for the blurred images
    """

    padx = 60
    pady = 60
    nose_x = 0
    nose_y = 0
    for idx, path in enumerate(frames):
        frame = Image.open(path)
        if nose_points[idx][0] != 0 and nose_points[idx][1] != 0:
            nose_x = nose_points[idx][0]
            nose_y = nose_points[idx][1]
            # print(nose_x, nose_y)

        elif left_ear_points[idx][0] != 0 and left_ear_points[idx][1] != 0:
            nose_x = left_ear_points[idx][0]
            nose_y = left_ear_points[idx][1]
        elif right_ear_points[idx][0] != 0 and right_ear_points[idx][1] != 0:
            nose_x = right_ear_points[idx][0]
            nose_y = right_ear_points[idx][1]
        else:
            # No point available - check previous point
            try:
                if nose_points[idx - 1][0] != 0 and nose_points[idx - 1][1] != 0:
                    nose_x = nose_points[idx - 1][0]
                    nose_y = nose_points[idx - 1][1]
                    # print(nose_x, nose_y)

                elif left_ear_points[idx - 1][0] != 0 and left_ear_points[idx - 1][1] != 0:
                    nose_x = left_ear_points[idx - 1][0]
                    nose_y = left_ear_points[idx][1]
                elif right_ear_points[idx - 1][0] != 0 and right_ear_points[idx - 1][1] != 0:
                    nose_x = right_ear_points[idx - 1][0]
                    nose_y = right_ear_points[idx - 1][1]
            except IndexError:
                nose_x = 0
                nose_y = 0

        point1 = nose_x - padx
        point2 = nose_y - pady
        if point1 < 0:
            point1 = 0
        if point2 < 0:
            point2 = 0

        nose = (int(point1), int(point2), int(nose_x + padx), int(nose_y + pady))
        cropped_frame = frame.crop(nose)
        blurred_frame = cropped_frame.filter(ImageFilter.GaussianBlur(radius=20))
        # print(nose)
        # sys.exit()
        frame.paste(blurred_frame, nose)

        outpath = "{}\\{}.png".format("blurred_coronal_images", idx + 1)
        print(outpath)
        frame.save(outpath)
    input_files = []
    for filename in glob.glob("{}\\*.png".format("blurred_coronal_images")):
        input_files.append(filename)
    # print(input_files)
    # Stupid python input_files.sort(key=lambda x: int(float(os.path.basename(x).split('.')[0][1:])))
    input_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    return input_files


def save_frame(frame):
    """
    Save a frame to output_images
    :param frame: Frame to save
    """
    time_stamp = datetime.now()
    filename = "{}.png".format(time_stamp.strftime("%Y-%m-%d_%H-%M-%S-%f"))
    path = "output_images\\{}".format(filename)
    cv2.imwrite(path, frame)


def image_to_gif():
    pass


def save_frame2(frame):
    """
    Save a frame to output_images
    :param frame: Frame to save
    """
    time_stamp = datetime.now()
    filename = "{}.png".format(time_stamp.strftime("%Y-%m-%d_%H-%M-%S-%f"))
    path = "output_coronal_images\\{}".format(filename)
    cv2.imwrite(path, frame)


def add_line_between_points(frame, points):
    """
    Adds a line overlay between two points and puts pixel distance text
    :param frame: Frame to add line over
    :param points: Two points to make line between
    :return: Processed frame with line over
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
    :return: Calculated distance (float)
    """
    print("ptt2", point1, point2)
    dist = np.linalg.norm(np.asarray(point1) - np.asarray(point2))
    print("Distance between two points: ", dist)
    return dist


def distance_overlay(display, point1, point2):
    """
    Creates overlay for distance (points and line)
    :param display: Frame to display over
    :param point1: First point
    :param point2: Second point
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
    # For mp4
    # _fourcc = 0x7634706d
    # For avi
    _fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if args['fps']:
        # video = cv2.VideoWriter("Output.mp4", _fourcc, args['fps'], (width, height))
        video = cv2.VideoWriter("Output.avi", _fourcc, args['fps'], (width, height))
    else:
        # video = cv2.VideoWriter("Output.mp4", _fourcc, 1, (width, height))
        video = cv2.VideoWriter("Output.avi", _fourcc, 1, (width, height))

    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


def save_video2():
    """
    Saves a video of processed image output to coronal processed_video directory
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
    """
    Return the magnitude of a point
    :param pt1: A 2D data point
    :return: float - magnitude of the point
    """
    return (pt1[0] ** 2 + pt1[1] ** 2) ** 0.5


def get_video_length(video_path):
    """
    Takes a video and return its length (s), frame count and fps
    :param video_path: path to video
    :return: length (seconds/float), frame count (int), fps (int)
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


def get_y_distance(pt1, pt2):
    """
    Return the y distance between two points
    :param pt1: A tuple containing a single x and y coordinate (x, y)
    :param pt2: A tuple containing a single x and y coordinate (x, y)
    :return: A float describing the absolute y displacement between the two points
    """
    distance = abs(pt1[1] - pt2[1])
    return distance


def resize_image(image, size_x, size_y, num):
    img = Image.open(image)
    new_img = img.resize((size_x, size_y))
    new_img.save("display_images/{}.png".format(num), "PNG", optimize=True)


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
