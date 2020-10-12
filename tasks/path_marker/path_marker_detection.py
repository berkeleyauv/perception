from combined_filter import init_combined_filter
from typing import Union

if __name__ == "__main__":
    import numpy as np
    import cv2
    from sys import argv as args

    # Data fron the new course footage dropbox folder
    # cap = cv2.VideoCapture('../data/course_footage/path_marker_GOPR1142.mp4')
    cap = cv2.VideoCapture(args[1])


def thresh_by_contour_size(
    frame: np.ndarray, num_contours: Union[int, None]
) -> np.ndarray:
    """ Assumes frame is grayscale

        Args:
            frame: The frame to analyze
            num_contours: The number of contours to return, sorted
                          by area from largest to smallest.
        Returns:
            frame/threshed: thresholded image or original image depending 
                on if num_contours specified.
    """

    frame = np.array(frame, np.uint8)

    img, contours, hierarchy = cv2.findContours(
        frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours is not None:
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        contours = contours[:num_contours]

        threshed = np.zeros(frame.shape, np.uint8)
        cv2.fillPoly(threshed, contours, 255)
        return threshed
    else:
        return frame


def find_path_marker(frame, draw_figs=False, thresh=0.3):
    """ Assumes frame is grayscale
        Returns angle of bottom line and top line relative to 0 radians
        This function doesn't guarantee that the angles are distinct
        Returns None if no good lines are found """

    def line_length(line):
        x0, y0, x1, y1 = line[0]
        return (x0 - x1) ** 2 + (y0 - y1) ** 2

    frame = thresh_by_contour_size(frame, num_contours=2)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame, 100, 150)

    # Find Hough lines
    # Source: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 2  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(
        edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )
    # lines[0] looks like [[x1, y1, x2, y2]]
    # where (x1, y1) is the first end point and (x2, y2) is the second end point

    if lines is not None:
        lines = lines.tolist()
        lines.sort(key=line_length, reverse=True)
        lines = lines[:10]

        # if line's y0 is below the average y0, it is a part of the top line, opposite for bottom line
        avgy = sum([l[0][1] + l[0][3] for l in lines]) // (len(lines) * 2)
        bot_lines = [l for l in lines if min(l[0][1], l[0][3]) > avgy]
        top_lines = [l for l in lines if max(l[0][1], l[0][3]) < avgy]

        if len(bot_lines) > 0 and len(top_lines) > 0:
            bot_angle = sum(
                [np.arctan2(l[0][1] - l[0][3], l[0][0] - l[0][2]) for l in bot_lines]
            ) / len(bot_lines)
            top_angle = sum(
                [np.arctan2(l[0][1] - l[0][3], l[0][0] - l[0][2]) for l in top_lines]
            ) / len(top_lines)

            if bot_angle - top_angle > np.pi:
                diff = (np.pi * 5 / 4 - (bot_angle - top_angle)) / 2
            else:
                diff = (np.pi * 3 / 4 - (bot_angle - top_angle)) / 2

            # abort if the detected marker segments are not close to 135 degrees apart at all
            if diff > thresh:
                if draw_figs:
                    cv2.imshow('lines bottom', frame)
                    cv2.imshow('lines top', frame)
                    cv2.imshow('frame with path marker angles', frame)
                return None

            # # otherwise, make the two segments 135 degrees apart
            # # Oh this might increase the weights given to bad data. meh
            # bot_angle += diff
            # top_angle -= diff

            if draw_figs:
                line_image_bot = frame.copy()
                line_image_top = frame.copy()
                for l in bot_lines:
                    cv2.line(line_image_bot, tuple(l[0][0:2]), tuple(l[0][2:4]), 255, 5)
                for l in top_lines:
                    cv2.line(line_image_top, tuple(l[0][0:2]), tuple(l[0][2:4]), 255, 5)

                cv2.imshow('lines bottom', line_image_bot)
                cv2.imshow('lines top', line_image_top)

                cv2.imshow(
                    'frame with path marker angles',
                    draw_marker_angles(frame, (bot_angle, top_angle)),
                )

            return bot_angle, top_angle
        else:
            if draw_figs:
                cv2.imshow('lines bottom', frame)
                cv2.imshow('lines top', frame)
                cv2.imshow('frame with path marker angles', frame)

            return None


def path_marker_get_new_heading(cap, is_approaching, draw_figs=False):
    """ Returns the next heading for the sub based on the path marker.
        (heading is positive for a counterclockwise turn)
        Takes an average of 10 frames.
        @param cap              a VideoCapture device, for example an .mp4 or a camera stream
        @param is_approaching   True: sub still wants to orient itself as it approaches
                                      the path marker. Returns the angle for the bottom leg of the
                                      path marker.
                                False: sub wants to orient itself towards wherever the path marker
                                       points towards. Returns angle for the top leg of the path marker.
    """
    angles = []
    test_angles = np.empty((10, 2))  # temporary. for testing porpoises

    # function aborts if the 10 most recent camera frames were invalid
    ret_tries = 0
    frames_used = 0

    while frames_used < 10 and ret_tries < 10:
        ret, frame = cap.read()
        if ret:
            ret_tries = 0
            frames_used += 1
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

            threshed = combined_filter(frame, False)
            new_angles = find_path_marker(threshed, draw_figs)

            if new_angles is not None:
                bot_angle, top_angle = new_angles

                if is_approaching:
                    angles.append(np.pi / 2 - bot_angle)
                    test_angles[len(angles) - 1] = (bot_angle, top_angle)
                else:
                    # top_angle is always negative so compare it to
                    # -np.pi/2
                    angles.append(-np.pi / 2 - top_angle)
                    test_angles[len(angles) - 1] = (bot_angle, top_angle)
            if draw_figs:
                cv2.waitKey(50)
        else:
            ret_tries += 1

    if (
        ret_tries >= 10 or len(angles) < 3
    ):  # does not return an answer if it was really unsure
        return None
    else:
        if draw_figs:
            cv2.imshow(
                'averaged angles',
                draw_marker_angles(
                    frame, test_angles[: len(angles)].sum(axis=0) / len(angles)
                ),
            )
        return sum(angles) / len(angles)


def draw_marker_angles(frame, marker_angles, right=False):
    """ Draws lines with the same angles as those in marker_angles off to
        the side of the frame """

    line_image = frame.copy()
    bot_angle, top_angle = marker_angles

    h, w = frame.shape[:2]
    if right:
        x, y = w * 0.25, h * 0.5
    else:
        x, y = w * 0.75, h * 0.5
    r = 20
    pt_mid = (int(x), int(y))
    pt_bot = (int(x + r * np.cos(bot_angle)), int(y + r * np.sin(bot_angle)))
    pt_top = (int(x + r * np.cos(top_angle)), int(y + r * np.sin(top_angle)))
    if frame.shape[2] == 1:
        line_image = cv2.line(line_image, pt_mid, pt_bot, 255, 5)
        line_image = cv2.line(line_image, pt_mid, pt_top, 255, 5)
    else:
        line_image = cv2.line(line_image, pt_mid, pt_bot, (255, 255, 255), 5)
        line_image = cv2.line(line_image, pt_mid, pt_top, (255, 255, 255), 5)

    return line_image


###########################################
# Main Body
###########################################

if __name__ == "__main__":
    marker_angles = None

    # # For testing purposes
    # # Thresholding is really bad if this starts at 650
    # for _ in range(600):
    #     cap.read()

    combined_filter = init_combined_filter()

    num_fails = 0
    while num_fails < 5:
        # Instead of passing in cap, we can alternatively pass in the last 10 frames as a list
        # but idk which is the better design choice
        new_heading = path_marker_get_new_heading(
            cap, is_approaching=True, draw_figs=True
        )
        print('new heading:', new_heading)

        if new_heading is None:
            num_fails += 1
        else:
            num_fails = 0

        k = cv2.waitKey(60) & 0xFF
        if k == 27:  # esc
            break

    cv2.destroyAllWindows()
    cap.release()
