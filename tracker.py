"""Marker detection with OpenCV"""

import cv2
import numpy as np
from marker import Marker

SQUARE_PX = 60
WIDTH = SQUARE_PX * 5
HEIGHT = SQUARE_PX * 5
CLOCKW_TRANSFORM = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
ACLOCKW_TRANSFORM = np.float32([[0, 0], [0, HEIGHT], [WIDTH, HEIGHT], [WIDTH, 0]])


def small_area(region):
    return cv2.contourArea(region) < 1e2


def not_quadrilateral(points):
    return len(points) != 4


def no_black_border(region):
    left = cv2.mean(region[0:60])
    right = cv2.mean(region[240:300])
    top = cv2.mean(region[:, 0:60])
    bottom = cv2.mean(region[:, 240:300])
    mean = np.mean(left + right + top + bottom)
    return mean > 10


def oriented_clockwise(polygon):
    x0 = polygon[0][0][0]
    y0 = polygon[0][0][1]
    x1 = polygon[1][0][0]
    y1 = polygon[1][0][1]
    x2 = polygon[2][0][0]
    y2 = polygon[2][0][1]
    cross = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
    return cross > 0


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

        if oriented_clockwise(polygon):
            trans_mat = CLOCKW_TRANSFORM
        else:
            trans_mat = ACLOCKW_TRANSFORM

        polygon_fl = np.float32(polygon)
        transform = cv2.getPerspectiveTransform(polygon_fl, trans_mat)
        sq_marker = cv2.warpPerspective(gray, transform, (WIDTH, HEIGHT))
        __, sq_marker_bin = cv2.threshold(sq_marker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if no_black_border(sq_marker_bin):
            continue

        marker = Marker.parse(sq_marker_bin)
        valid_marker, marker_id, rotations = Marker.validate(marker)

        if not valid_marker:
            continue

        if not with_id:
            markers.append(Marker(marker_id, contour, polygon, rotations))
        elif with_id == marker_id:
            return Marker(marker_id, contour, polygon, rotations)

    return markers


def find_marker_with_id(img, with_id):
    return find_markers(img, with_id)