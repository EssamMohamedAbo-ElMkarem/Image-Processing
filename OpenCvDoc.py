#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:42:58 2021

@author: xmagneto(Essam Mohamed)
"""
xes = list()
yes = list()
colors = list()
widths = list()


    

class cvMain(object):
    def __init__(self):
        cv2.setUseOptimized(True)
        self.c = int(input("1- Color Mixer   2- Color Filter\n3-Drawer         4- Thresholder\n5- PrespectiveTransform      6- LPF\n7- Morpho      8- EdgeDetector\n9- Template Matching    10- Line Detection\n11- ContourDetection\n12- Harris CornerDet 99-Exit\n>>>"))
        if self.c == 1:
            self.mx = self.runVidColorMixer(0, 1)
        elif self.c == 2:
            self.mx = self.runVidColorFilter(0, 1)
        elif self.c == 3:
            self.dr = Drawer()
        elif self.c == 4:
            self.th = self.imageThresholding(0, 1)
        elif self.c == 5:
            self.pt = self.prespectiveTransform()
        elif self.c == 6:
            self.lpf = self.LPF()
        elif self.c == 7:
            self.m = self.morphoLogicalTransforms()
        elif self.c == 8:
            self.ed = self.edgeDetection()
        elif self.c == 9:
            self.tm = self.templateMatching()
        elif self.c == 10:
            self.ld = self.lineDEtection()
        elif self.c == 11:
            self.cd = self.contourDetection()
        elif self.c == 12:
            self.har = self.harris()
        elif self.c == 99:
            print("Thanks for using our script")
            sys.exit(0)
        else:
            print("Please, Enter a valid value")
            self.main = cvMain()

    def nothing(self, x):
        pass
    
    def runVidColorMixer(self, path, wt):
        cap = cv2.VideoCapture(path)
        cv2.namedWindow('image')
        cv2.createTrackbar('R','image',0,255,self.nothing)
        cv2.createTrackbar('G','image',0,255,self.nothing)
        cv2.createTrackbar('B','image',0,255,self.nothing)
        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'image',0,1,self.nothing)
        
        while True:
            _, frame = cap.read()
            
            r = cv2.getTrackbarPos('R','image')
            g = cv2.getTrackbarPos('G','image')
            b = cv2.getTrackbarPos('B','image')
            s = cv2.getTrackbarPos(switch,'image')
            
            (B, G, R) = cv2.split(frame)
            B += b; G += g; R+=r
            merged = cv2.merge([B, G, R])
            if s == 0:
                img = frame
            else:
                img = merged
            cv2.imshow('image', img)
            
            wk = cv2.waitKey(wt)
            if wk == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        
    def runVidColorFilter(self, path, wt):
        cap = cv2.VideoCapture(path)
        cv2.namedWindow('image')
        cv2.createTrackbar('H_lower','image',0,255,self.nothing)
        cv2.createTrackbar('S_lower','image',0,255,self.nothing)
        cv2.createTrackbar('V_lower','image',0,255,self.nothing)
        cv2.createTrackbar('H_higher','image',0,255,self.nothing)
        cv2.createTrackbar('S_higher','image',0,255,self.nothing)
        cv2.createTrackbar('V_higher','image',0,255,self.nothing)
        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'image',0,1,self.nothing)
        
        while True:
            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            h_l = cv2.getTrackbarPos('H_lower','image')
            s_l = cv2.getTrackbarPos('S_lower','image')
            v_l = cv2.getTrackbarPos('V_lower','image')
            h_h = cv2.getTrackbarPos('H_higher','image')
            s_h = cv2.getTrackbarPos('S_higher','image')
            v_h = cv2.getTrackbarPos('V_higher','image')
            s = cv2.getTrackbarPos(switch,'image')
            
            lower_hsv = np.array([h_l, s_l, v_l])
            upper_hsv = np.array([h_h, s_h, v_h])
        
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            result = cv2.bitwise_and(frame, frame, mask = mask)
            
            if s == 0:
                img = frame
            else:
                img = result
            cv2.imshow('image', img)
            
            wk = cv2.waitKey(wt)
            if wk == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def imageThresholding(self, path, wt):
        cap = cv2.VideoCapture(path)
        cv2.namedWindow('image')
        cv2.createTrackbar('Threshold','image',0,255,self.nothing)
        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'image',0,1,self.nothing)
        
        while True:
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            s = cv2.getTrackbarPos(switch,'image')
            threshold = cv2.getTrackbarPos('Threshold','image')
            ret, thresh1 = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
            ret, thresh2 = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)
            thresh3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            thresh4 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            if s == 0:
                cv2.imshow('original', frame)
            else:
                cv2.imshow("BinaryThresh", thresh1)
                cv2.imshow("InvBinaryThresh", thresh2)
                cv2.imshow("MeanAdaptiveThresh", thresh3)
                cv2.imshow("GaussianMeanThresh", thresh4)
            
            wk = cv2.waitKey(wt)
            if wk == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        
    def prespectiveTransform(self):
       
        img = cv2.imread('drawing.png')
        rows,cols,ch = img.shape
        pts1 = np.float32([[59,69],[455,55],[28,478],[480,480]])/2
        pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])/2
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(150,150))
        plt.subplot(121),plt.imshow(img),plt.title('Input')
        plt.subplot(122),plt.imshow(dst),plt.title('Output')
        plt.show()
        
    def LPF(self):
        
        img = cv2.imread("logo.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        av_blur = cv2.blur(img,(75, 75))
        gs_blur = cv2.GaussianBlur(img, (5, 5), 0)
        median = cv2.medianBlur(img, 5)
        bf = cv2.bilateralFilter(img, 9, 75, 75)
        plt.subplot(221),plt.imshow(av_blur),plt.title('AverageBlurred')
        plt.xticks([]), plt.yticks([])
        plt.subplot(222),plt.imshow(gs_blur),plt.title('GaussianBlurred')
        plt.xticks([]), plt.yticks([])
        plt.subplot(223),plt.imshow(median),plt.title('MedianBlurred')
        plt.xticks([]), plt.yticks([])
        plt.subplot(224),plt.imshow(bf),plt.title('BilateralFilter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    def morphoLogicalTransforms(self):
        img = cv2.imread("dil.png")
        close = cv2.imread("close.jpeg")
        op = cv2.imread("open.jpeg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        close = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
        op = cv2.cvtColor(op, cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(close, 127, 255, cv2.THRESH_BINARY)
        _, thresh3 = cv2.threshold(op, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        cv2.imshow("original", img)
        cv2.imshow("Gradient", cv2.morphologyEx(thresh3, cv2.MORPH_GRADIENT, kernel))
        cv2.imshow("Erosion", cv2.erode(thresh1, kernel, iterations=1))
        cv2.imshow("Dialation", cv2.dilate(thresh1, kernel, iterations=1))
        cv2.imshow("Opening", cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel))
        cv2.imshow("Closing", cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def edgeDetection(self):
        img = cv2.imread('sudo.jpg', 0)
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        edges = cv2.Canny(cv2.imread("patella.png"),90,250)
        cv2.imshow("laplacian", laplacian)
        cv2.imshow("x", sobelx)
        cv2.imshow("y", sobely)
        cv2.imshow("Canny", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def templateMatching(self):
        img = cv2.imread('anne.jpg',0)
        img2 = img.copy()
        template = cv2.imread('face.jpg',0)
        w, h = template.shape[::-1]
        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                   'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF_NORMED']
        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            # Apply template Matching
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img,top_left, bottom_right, (255, 255, 255), 10)
            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()
            
    def lineDEtection(self):
        img = cv2.imread('road.png')
        imgp = img.copy()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/180,250)
        
        minLineLength = 200
        maxLineGap = 10
        linesp = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
        print(linesp)
        for line in linesp:
            for x1,y1,x2,y2 in line:
                cv2.line(imgp,(x1,y1),(x2,y2),(0,255,0),2)
        print(lines)
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow('Cany',edges)
        cv2.imshow('Hough',img)
        cv2.imshow('HoughP', imgp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def contourDetection(self):
        img = cv2.imread('cnt.webp')
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            epsilon = 0.01*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            print(approx)
            cv2.drawContours(img, [approx], 0, (0), 3)
            # Position for writing text
            x,y = approx[0][0]
            
            if len(approx) == 3:
                cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            elif len(approx) == 4:
                cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            elif len(approx) == 5:
                cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            elif 6 < len(approx) < 15:
                cv2.putText(img, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            else:
                cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
        cv2.imshow("final", img)
        cv2.waitKey(0)
    def harris(self):
        img = cv2.imread("raf.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        res = cv2.cornerHarris(gray, 2, 3, 0.04)
        img[res > 0.01*res.max()] = (0, 0, 255)
        cv2.imshow("corners", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()            
        
        
        
        
class Drawer(object):

    def __init__(self):
        self.root = Tk()
        self.root.title("PyFrame")

        self.menu_bar = Menu(self.root)
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="New", command=Drawer)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.destroy)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.settings = Menu(self.menu_bar, tearoff=0)
        self.settings.add_command(label="Frame BG color", command=self.bg_frame)
        self.settings.add_command(label="PenWidth", command=PenWidth)
        self.menu_bar.add_cascade(label="Settings", menu=self.settings)
        self.delete_menu = Menu(self.menu_bar, tearoff=0)
        self.delete_menu.add_command(label="Clear", command=self.clear)
        self.menu_bar.add_cascade(label="Procedures", menu=self.delete_menu)
        self.help_menu = Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=Drawer.about)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.root.config(menu=self.menu_bar)

        self.frame = Frame(self.root, width=590, height=500)
        self.frame.pack()
        self.interact_frame = Frame(self.root)
        self.interact_frame.pack()
        self.can = Canvas(self.frame, width=590, height=500, bg="dark green")
        self.can.pack()
        self.can.bind("<Button-1>", self.event)
        self.lbl = Label(self.interact_frame, fg="dark green")
        self.lbl.grid(row=1, column=1)
        self.lin = Button(self.interact_frame, relief=FLAT, text="CreateLine", fg="white", bg="royal blue", command=self.create_line)
        self.lin.grid(row=2, column=0, padx=60)
        self.ovl = Button(self.interact_frame, relief=FLAT, text="CreateOval", fg="white", bg="royal blue", command=self.create_oval)
        self.ovl.grid(row=2, column=1, padx=60)
        self.color = Button(self.interact_frame, relief=FLAT, fg="white", bg="royal blue", text="ShapeColor", command=Drawer.color)
        self.color.grid(row=2, column=2, padx=60)
        self.pol = Button(self.interact_frame, relief=FLAT, text="CreatePoly", fg="white", bg="royal blue", command=self.create_poly)
        self.pol.grid(row=3, column=0, padx=60, pady=3)
        self.arc = Button(self.interact_frame, relief=FLAT, text="Create_Arc", fg="white", bg="royal blue", command=self.create_arc)
        self.arc.grid(row=3, column=1, padx=60, pady=3)
        self.rec = Button(self.interact_frame, relief=FLAT, text="CreateRect", fg="white", bg="royal blue", command=self.create_rec)
        self.rec.grid(row=3, column=2, padx=60, pady=3)
        self.root.mainloop()

    @staticmethod
    def color():
        selected_color = tkinter.colorchooser.askcolor()
        colors.append(selected_color[1])
    @staticmethod
    def about():
        tkinter.messagebox.showinfo("About", "I have been created by Essam Mohamed")
    
    def clear(self):
        self.can.create_rectangle(0, 0, 591, 501, fill="dark green")

    def bg_frame(self):
        bg_color = tkinter.colorchooser.askcolor()
        self.can.configure(bg=bg_color[1])

    def event(self, event):

        self.lbl.configure(text="X of the click was " + str(event.x) + " Y of the click was " + str(event.y))

        try:

            xes.append(event.x)
            yes.append(event.y)
            global x1
            x1 = xes[len(xes) - 2]
            global x2
            x2 = xes[len(xes) - 1]
            global x3
            x3 = xes[len(xes) - 3]
            global y1
            y1 = yes[len(yes) - 2]
            global y2
            y2 = yes[len(yes) - 1]
            global y3
            y3 = yes[len(yes) - 3]

        except IndexError:
            pass

    def create_line(self):
        try:
            self.can.create_line(x1, y1, x2, y2, width=widths[len(widths) - 1], fill=colors[len(colors) - 1])
        except IndexError:
            tkinter.messagebox.showwarning("", "Color hasn't been chosen or width hasn't been given a value")

    def create_oval(self):
        try:
            self.can.create_oval(x1, y1, x2, y2, width=widths[len(widths) - 1], fill=colors[len(colors) - 1])
        except IndexError:
            tkinter.messagebox.showwarning("", "Color hasn't been chosen or width hasn't been given a value")

    def create_poly(self):
        try:
            self.can.create_polygon(x1, y1, x2, y2, x3, y3, width=widths[len(widths) - 1], fill=colors[len(colors) - 1])
        except IndexError:
            tkinter.messagebox.showwarning("", "Color hasn't been chosen or width hasn't been given a value")

    def create_arc(self):
        try:
            self.can.create_arc(x1, y1, x2, y2, width=widths[len(widths) - 1], fill=colors[len(colors) - 1])
        except IndexError:
            tkinter.messagebox.showwarning("", "Color hasn't been chosen or width hasn't been given a value")

    def create_rec(self):
        try:
            self.can.create_rectangle(x1, y1, x2, y2, width=widths[len(widths) - 1], fill=colors[len(colors) - 1])
        except IndexError:
            tkinter.messagebox.showwarning("", "Color hasn't been chosen or width hasn't been given a value")


class PenWidth:
    def __init__(self):
        self.root = Tk()
        self.root.title("Width")
        self.root.maxsize(180, 70)
        self.root.minsize(180, 70)
        self.lbl = Label(self.root, text="Enter pen's width in pxs...")
        self.lbl.pack()
        self.w = Entry(self.root, text="2")
        self.w.pack()
        self.b = Button(self.root, text="Set", bg="royal blue", fg="white", command=self.set_width)
        self.b.pack()
        self.root.mainloop()

    def set_width(self):
        widths.append(self.w.get())



if __name__ == '__main__':
    import cv2
    import sys
    import numpy as np
    from matplotlib import pyplot as plt
    from tkinter import *
    import tkinter.messagebox
    import tkinter.colorchooser
    app = cvMain()