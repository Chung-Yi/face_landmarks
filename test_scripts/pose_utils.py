import numpy as np
import cv2


class Annotator():
    def __init__(self,
                 im,
                 angles=None,
                 lm=None,
                 rvec=None,
                 tvec=None,
                 cm=None,
                 dc=None,
                 b=10.0):
        self.im = im

        self.angles = angles
        self.bbox = bbox
        self.lm = lm
        self.rvec = rvec
        self.tvec = tvec
        self.cm = cm
        self.dc = dc
        self.nose = tuple(lm[0].astype(int))
        self.box = np.array([(b, b, b), (b, b, -b), (b, -b, -b), (b, -b, b),
                             (-b, b, b), (-b, b, -b), (-b, -b, -b), (-b, -b,
                                                                     b)])
        self.b = b

    def draw_direction(self):
        (nose_end_point2D, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, self.b)]), self.rvec, self.tvec, self.cm,
            self.dc)
        p1 = self.nose
        p2 = tuple(nose_end_point2D[0, 0].astype(int))
        cv2.line(self.im, p1, p2, Color.yellow, self.ls)
        return self.im
