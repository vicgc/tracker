import cv2
import numpy as np


class Marker:
    """Marker class which gets designated to all valid markers"""

    def __init__(self, marker_id, contour, polygon, rotations=0):
        self.id = marker_id
        self.contour = contour
        self.polygon = polygon
        self.rotations = rotations

        self.cx, self.cy = self.position
        self.x, self.y = self.corners

    @property
    def position(self):
        m = cv2.moments(self.contour)
        x = int(m['m10']/m['m00'])
        y = int(m['m01']/m['m00'])
        return x, y

    @property
    def corners(self):
        x, y = np.hsplit(np.squeeze(self.polygon), 2)
        x, y = map(np.squeeze, [x, y])
        return x, y

    @property
    def major_axis(self):
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