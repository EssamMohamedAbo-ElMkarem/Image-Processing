class ShapeDetection(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            _, self.frame = self.cap.read()
            self.detect_shape_contour_approx(self.frame)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()
        self.cap.release()

    @staticmethod
    def detect_shape_contour_approx(image):
        font = cv2.FONT_HERSHEY_COMPLEX
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(img, [approx], 0, 0, 5)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if len(approx) == 4:
                x1, y1, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 < aspect_ratio < 1.05:
                    cv2.putText(image, "SQUARE" + str(len(approx)), (x, y), font, 1, (0, 255, 255))
                else:
                    cv2.putText(image, "RECTANGLE" + str(len(approx)), (x, y), font, 1, (255, 0, 0))

            elif len(approx) == 3:
                cv2.putText(image, "TRIANGLE" + str(len(approx)), (x, y), font, 1, (255, 0, 0))
            else:
                cv2.putText(image, "CIRCLE" + str(len(approx)), (x, y), font, 1, (0, 255, 0))
        cv2.imshow("Shapes", image)
        cv2.imshow("Threshold", threshold)


if __name__ == "__main__":
    import cv2
    app = ShapeDetection()