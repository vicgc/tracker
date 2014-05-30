"""Marker detection with OpenCV"""

import cv2
import numpy as np
from marker import Marker

SQUARE_PX = 60
WIDTH = SQUARE_PX * 5
HEIGHT = SQUARE_PX * 5

VALID_MARKERS = {
    1: [[1, 0, 1], [0, 0, 0], [0, 0, 1]],
    2: [[1, 0, 1], [0, 0, 1], [0, 0, 1]],
    3: [[1, 0, 1], [0, 0, 0], [0, 1, 1]],
    4: [[1, 1, 1], [0, 0, 0], [0, 0, 1]],
    5: [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
    6: [[1, 1, 1], [0, 0, 0], [0, 1, 1]]
}


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
    x, y = map(np.squeeze, np.hsplit(np.squeeze(polygon), 2))
    cross = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
    return cross > 0


def transform_matrix(polygon):
    if oriented_clockwise(polygon):
        return np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    else:
        return np.float32([[0, 0], [0, HEIGHT], [WIDTH, HEIGHT], [WIDTH, 0]])


def parse_marker(marker):
    marker_data = np.zeros(shape=(3, 3), dtype=np.int)

    for i, x in enumerate(xrange(60, 181, 60)):
        for j, y in enumerate(xrange(60, 181, 60)):
            if np.mean(marker[x:x+60, y:y+60]) > 200:
                marker_data[i, j] = 1

    return marker_data


def validate_marker(marker):
    for marker_id, valid_marker in VALID_MARKERS.iteritems():
        for rotations in xrange(4):
            if np.array_equal(marker, np.rot90(valid_marker, rotations)):
                return True, marker_id, rotations

    return False, None, None


def find_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    __, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, __ = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)

    markers = dict()

    for contour in contours:
        if small_area(contour):
            continue

        eps = 0.05 * cv2.arcLength(contour, closed=True)
        polygon = cv2.approxPolyDP(contour, eps, closed=True)

        if not_quadrilateral(polygon):
            continue

        polygon_fl = np.float32(polygon)
        tr_matrix = transform_matrix(polygon)
        transform = cv2.getPerspectiveTransform(polygon_fl, tr_matrix)
        sq_marker = cv2.warpPerspective(gray, transform, (WIDTH, HEIGHT))
        __, sq_marker_bin = cv2.threshold(sq_marker, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if no_black_border(sq_marker_bin):
            continue

        marker = parse_marker(sq_marker_bin)
        valid_marker, marker_id, rotations = validate_marker(marker)

        if not valid_marker:
            continue

        markers[marker_id] = Marker(marker_id, contour, polygon, rotations)

    return markers