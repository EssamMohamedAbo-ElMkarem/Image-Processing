class Live_KMeans(object):

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        while True:
            _, self.frame = self.cap.read()
            self.kmeans(self.frame, 3)
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
        cv2.destroyAllWindows()
        self.cap.release()

    @staticmethod
    def kmeans(image, k=3):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        whit_idx = np.argmin(np.abs(np.sum(centers - whit, axis=1)))
        pink_idx = np.argmin(np.abs(np.sum(centers - pink, axis=1)))
        new_centers = np.zeros((10, 3), dtype=np.uint8)
        new_centers[whit_idx] = whit
        new_centers[pink_idx] = pink

        labels = labels.flatten()
        segmented_image = new_centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        return segmented_image

if __name__ == "__main__":
    import cv2
    import numpy as np
    import imutils
    app = Live_KMeans()
