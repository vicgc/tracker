"""Marker detection with OpenCV"""

import cv2
import numpy as np
from marker import Marker

SQUARE_PX = 60
WIDTH = SQUARE_PX * 5
HEIGHT = SQUARE_PX * 5

TRANSFORM = {'clockwise': np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]),
             'aclockwise': np.float32([[0, 0], [0, HEIGHT], [WIDTH, HEIGHT], [WIDTH, 0]])}

VALID_MARKERS = [
    [[1, 0, 1], [0, 0, 0], [0, 0, 1]],
    [[1, 0, 1], [0, 0, 1], [0, 0, 1]],
    [[1, 0, 1], [0, 0, 0], [0, 1, 1]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 1]],
    [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
    [[1, 1, 1], [0, 0, 0], [0, 1, 1]]
]

MARKER_ID = [1, 2, 3, 4, 5, 6]


def small_area(region):
    return cv2.contourArea(region) < 1e2


def not_quadrilateral(points):
    return len(points) != 4


def no_black_border(region):
    left = cv2.mean(region[0:60])
    right = cv2.mean(region[240:300])
    top = cv2.mean(region[60:240, 0:60])
    bottom = cv2.mean(region[60:240, 240:300])
    mean = np.mean(left + right + top + bottom)
    return mean > 10


def oriented_clockwise(polygon):
    x, y = np.hsplit(np.squeeze(polygon), 2)
    x, y = map(np.squeeze, [x, y])
    cross = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
    return cross > 0


def parse_marker(marker):
    marker_data = np.zeros(shape=(3, 3), dtype=np.int)

    for i, x in enumerate(range(60, 240, 60)):
        for j, y in enumerate(range(60, 240, 60)):
            mean = np.mean(marker[x:x+60, y:y+60])
            if mean > 200:
                marker_data[i, j] = 1

    return marker_data


def validate_marker(marker):
    for i, valid_marker in enumerate(VALID_MARKERS):
        for rotations in xrange(4):
            if (marker == np.rot90(valid_marker, rotations)).all():
                return True, MARKER_ID[i], rotations

    return False, None, None


def find_markers(img, with_id=False):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    __, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, __ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    markers = list()

    for contour in contours:

        if small_area(contour):
            continue

        eps = 0.05 * cv2.arcLength(contour, closed=True)
        polygon = cv2.approxPolyDP(contour, eps, closed=True)

        if not_quadrilateral(polygon):
            continue

        orientation = 'clockwise' if oriented_clockwise(polygon) else 'aclockwise'
        polygon_fl = np.float32(polygon)
        transform = cv2.getPerspectiveTransform(polygon_fl, TRANSFORM[orientation])
        sq_marker = cv2.warpPerspective(gray, transform, (WIDTH, HEIGHT))
        __, sq_marker_bin = cv2.threshold(sq_marker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if no_black_border(sq_marker_bin):
            continue

        marker = parse_marker(sq_marker_bin)
        valid_marker, marker_id, rotations = validate_marker(marker)

        if not valid_marker:
            continue

        if not with_id:
            markers.append(Marker(marker_id, contour, polygon, rotations))
        elif with_id == marker_id:
            return Marker(marker_id, contour, polygon, rotations)

    return markers


def find_marker_with_id(img, with_id):
    return find_markers(img, with_id)