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
        f = lambda z: z[(4-r) % 4] + int((z[(5-r) % 4] - z[(4-r) % 4]) / 2)
        x, y = map(f, [self.x, self.y])
        return x, y

    @property
    def minor_axis(self):
        r = self.rotations
        f = lambda z: z[(5-r) % 4] + int((z[(6-r) % 4] - z[(5-r) % 4]) / 2)
        x, y = map(f, [self.x, self.y])
        return x, y

    @property
    def area_vec(self):
        pass

    def angle_to_point(self, point):
        a, b, c = map(np.array, [self.major_axis, self.position, point])

        phi = np.arctan2(*(a - b))
        rho = np.arctan2(*(c - b))

        if phi < 0:
            phi += 2*np.pi

        if rho < 0:
            rho += 2*np.pi

        angle = round(np.degrees(rho - phi))

        return angle