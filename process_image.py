import cv2
import numpy as np

gamma = 0.95

Wcb = 46.97
Wcr = 38.76

WHCb = 14
WHCr = 10
WLCb = 23
WLCr = 20

Ymin = 16
Ymax = 235

Kl = 125
Kh = 188

WCb = 0
WCr = 0

CbCenter = 0
CrCenter = 0


class Image:
    def __init__(self):
        self.lower = np.array([0, 40, 80], dtype='uint8')
        self.upper = np.array([20, 255, 255], dtype='uint8')

    def skin_detect(face_image):
        rows = face_image.shape[0]
        cols = face_image.shape[1]

        for r in range(rows):
            for c in range(cols):
                # get values of blue, green, red
                B = face_image.item(r, c, 0)
                G = face_image.item(r, c, 1)
                R = face_image.item(r, c, 2)
                # gamma correction
                B = int(B**gamma)
                G = int(G**gamma)
                R = int(R**gamma)

                # set values of blue, green, red
                face_image.itemset((r, c, 0), B)
                face_image.itemset((r, c, 1), G)
                face_image.itemset((r, c, 2), R)

        # convert color space from rgb to ycbcr
        imgYcc = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCR_CB)

        # convert color space from bgr to rgb
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # prepare an empty image space
        imgSkin = np.zeros(img.shape, np.uint8)
        # copy original image
        imgSkin = img.copy()

        for r in range(rows):
            for c in range(cols):
                skin = 0

                # color space transformation

                # get values from ycbcr color space
                Y = imgYcc.item(r, c, 0)
                Cr = imgYcc.item(r, c, 1)
                Cb = imgYcc.item(r, c, 2)

                if Y < Kl:
                    WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
                    WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)

                    CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
                    CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)

                elif Y > Kh:
                    WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
                    WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)

                    CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
                    CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)

                if Y < Kl or Y > Kh:
                    Cr = (Cr - CrCenter) * Wcr / WCr + 154
                    Cb = (Cb - CbCenter) * Wcb / WCb + 108

                if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
                    skin = 1

                if 0 == skin:
                    imgSkin.itemset((r, c, 0), 0)
                    imgSkin.itemset((r, c, 1), 0)
                    imgSkin.itemset((r, c, 2), 0)
        return img, imgSkin

    def detect(self, face_image):
        converted = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        skinmask = cv2.inRange(converted, self.lower, self.upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        skinmask = cv2.erode(skinmask, kernel, iterations=2)
        skinmask = cv2.dilate(skinmask, kernel, iterations=2)

        skinmask = cv2.GaussianBlur(skinmask, (3, 3), 0)
        skin = cv2.bitwise_and(face_image, face_image, mask=skinmask)
        return skin