class FlexibleFilter(object):

    def __init__(self, lower_color_arr1, upper_color_arr1, lower_color_arr2, upper_color_arr2):
        cap = cv2.VideoCapture(0)
        while True:
            _, self.frame = cap.read()
            self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            self.lower_color1 = np.array(lower_color_arr1)
            self.upper_color1 = np.array(upper_color_arr1)
            self.lower_color2 = np.array(lower_color_arr2)
            self.upper_color2 = np.array(upper_color_arr2)
            self.mask1 = cv2.inRange(self.hsv, self.lower_color1, self.upper_color1)
            self.mask2 = cv2.inRange(self.hsv, self.lower_color2, self.upper_color2)
            self.mask = self.mask1 | self.mask2
            self.res = cv2.bitwise_and(self.frame, self.frame, mask=self.mask)
            plt.imshow('frame', self.frame)
            plt.imshow('filtered result', self.res)
            plt.show()
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()
        cap.release()

        
if __name__ == "__main__":

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    FlexibleFilter([], [], [], []) # values of filtered colors' ranges