"""Marker class which gets designated to all valid markers"""

import cv2
import numpy as np

VALID_MARKERS = [
    [[1, 0, 1], [0, 0, 0], [0, 0, 1]],
    [[1, 0, 1], [0, 0, 1], [0, 0, 1]],
    [[1, 0, 1], [0, 0, 0], [0, 1, 1]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 1]],
    [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
    [[1, 1, 1], [0, 0, 0], [0, 1, 1]]
]

MARKER_ID = [1, 2, 3, 4, 5, 6]


class Marker:
    def __init__(self, marker_id, contour, polygon, rotations=0):
        self.id = marker_id
        self.contour = contour
        self.polygon = polygon
        self.rotations = rotations

        self.position = self.__pos()
        self.cx, self.cy = self.position
        self.x, self.y = self.__corners()
        self.major_axis = self.__major_axis()

    def __pos(self):
        moments = cv2.moments(self.contour)
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        return cx, cy

    def __corners(self):
        x, y = list(), list()
        for i in xrange(4):
            x.append(self.polygon[i][0][0])
            y.append(self.polygon[i][0][1])
        return x, y

    def __major_axis(self):
        r = self.rotations
        x = self.x[(4-r) % 4] + int((self.x[(5-r) % 4] - self.x[(4-r) % 4]) / 2)
        y = self.y[(4-r) % 4] + int((self.y[(5-r) % 4] - self.y[(4-r) % 4]) / 2)
        return x, y

    def angle_to_point(self, point):
        a = np.array(self.major_axis)
        b = np.array(self.position)
        c = np.array(point)

        phi = np.arctan2(*(a - b))
        if phi < 0:
            phi += 2*np.pi

        rho = np.arctan2(*(c - b))
        if rho < 0:
            rho += 2*np.pi

        return round(np.degrees(rho - phi))

    @staticmethod
    def parse(marker):
        marker_data = np.zeros(shape=(3, 3), dtype=np.int)

        # perhaps rewrite this to check for avg. color
        for i, x in enumerate(range(90, 240, 60)):
            for j, y in enumerate(range(90, 240, 60)):
                if marker[x, y] == 255:
                    marker_data[i, j] = 1

        return marker_data

    @staticmethod
    def validate(marker):
        for i, valid_marker in enumerate(VALID_MARKERS):
            for rotations in xrange(4):
                if (marker == np.rot90(valid_marker, rotations)).all():
                    return True, MARKER_ID[i], rotations

        return False, None, None
