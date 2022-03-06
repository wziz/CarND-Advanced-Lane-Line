import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import glob
from scipy.interpolate import CubicSpline
import copy

#  camera calibration
images = glob.glob('camera_cal/calibration*.jpg')
imgpoints = []  # list to store image points of all pictures
objpoints = []  # list to store object points of all pictures
#  prepare object points like (0, 0, 0), ..., (5, 8, 0)
objp = np.zeros((54, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)
for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale
    ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)  # find chessboard corners
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
img = mpimg.imread(images[0])
_, mtx_calc, dist_calc, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

# transform Perspective
img = mpimg.imread('test_images/test2.jpg')
img_size = (img.shape[1], img.shape[0])
scr = np.float32([[575, 465], [(1280/2-575)+1280/2, 465], [1280/2-264+1280/2, 680], [264, 680]])
dst = np.float32([[350, 100], [640-350+640, 100], [640-350+640, 680], [350, 680]])
M = cv2.getPerspectiveTransform(scr, dst)
Minv = cv2.getPerspectiveTransform(dst, scr)


class frame_class(object):  # class, which is to processing every frame
    def __init__(self):
        self.image = []
        self.counter = 0
        self.left_fitx_mem = []
        self.right_fitx_mem = []
        self.left_fit = None
        self.right_fit = None
        self.ym_per_pix = 3 / (678 - 583 + 1)
        self.xm_per_pix = 3.7 / (944 - 363 + 1)
        self.frame_drop = False
        self.save = []
        # define the curve for adjusting a color-image (used in method: self.ps_curve_proc())
        self.points = np.array([0, 20, 55, 90, 160, 200, 230, 255])
        self.values = np.array([0, 10, 50, 90, 160, 220, 250, 255])
        self.cs = CubicSpline(self.points, self.values)
        # define the curve for adjusting a gray-image (used in method: self.ps_curve_proc())
        self.points_gray = np.array([0,40,255])
        self.values_gray = np.array([0, 0, 255])
        self.cs_gray = CubicSpline(self.points_gray, self.values_gray)
        self.method = ''
    #  useful functions
    def grayshow(self, image, t=None): # for debug, plot image as gray picture
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(t)
    def output_mode(self, select_region, mode): # select, in which form will the selected region output: 0/1? True/False? ploted or not?
        if mode == 1:  # simply return the "select_region"
            output = select_region
        elif mode == 2 or mode == 3:
            binary_output = np.zeros_like(select_region)
            binary_output[select_region] = 1
            if mode == 2:  # return the "select_region" as 0/1 for further use and plot the selected region in gray
                output = binary_output
                plt.figure()
                plt.imshow(binary_output, cmap='gray')
            else:  # return the "select_region" as 0/1 and True/False
                output = [binary_output, select_region]
        return output
    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0,255), mode=1):  # absolute sobel threshhold
        if len(image.shape) > 2:  # in case of color image
            print('image must be a one channel image')
            sys.exit()
        #  compute gradient
        if orient == 'x':
            sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        select_region = ((scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1]))
        output = self.output_mode(select_region, mode)
        return output
    def mag_thresh(self, image, sobel_kernel= 3, mag_thresh=(0,255), mode=1):  # threshholding of magnitude of gradient
        if len(image.shape) > 2:  # in case of color image
            print('image must be a one channel image')
            sys.exit()
        #  compute magnitude
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        magnitude = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
        scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        select_region = ((scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude <= mag_thresh[1]))
        output = self.output_mode(select_region, mode)
        return output
    def dir_thresh(self, image, sobel_kernel=3, thresh=(0, np.pi/2), mode=1):  # threshholding of direction of gradient
        if len(image.shape) > 2:  # in case of color image
            print('image must be a one channel image')
            sys.exit()
        #  compute direction of gradient
        abs_x = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_y = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        direction = np.arctan2(abs_y, abs_x)
        select_region = ((direction >= thresh[0]) & (direction <= thresh[1]))
        output = self.output_mode(select_region, mode)
        return output
    def color_thresh(self, image, thresh=(0,255), mode=1): # threshholding one color channel (gray, r, g, b, ...) of a image
        select_region = ((image >= thresh[0]) & (image <= thresh[1]))
        output = self.output_mode(select_region, mode)
        return output
    def cor_dist(self, image, mtx=mtx_calc, dist=dist_calc): # distortion Correction
        undst = cv2.undistort(image, mtx, dist, None, mtx)
        return undst
    def getZValueFromTabel(self, input_x, tiFil_tabel): # get the value from lookup tabel. In this project it is used to get the filter constant in reponding to difference of two frames for a stable lane line detection
        i = 0
        while input_x > tiFil_tabel[0][i] and i < len(tiFil_tabel[0]) - 1:
            i += 1
        return tiFil_tabel[1][i]
    def contrast_adj(self, image, contrast): # to adjust the contrast of a image for better lane line detection
        buf = image.copy()
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        return buf
    def ps_curve_proc(self, input_img, mode='color'): # realize the function of Photoshop curve. In this project it is used to fine tune the range and the transition of brightness and shallow of the image
        if mode == 'color':
            cs = self.cs  # curve for adjusting a color-image
        elif mode == 'gray':
            cs = self.cs_gray # curve for adjusting a gray-image
        else:
            cs = self.cs
        output_img = input_img.copy()
        output_img = cs(output_img)
        output_img[output_img > 255] = 255
        output_img = output_img.astype(np.uint8)
        def plot_ps_curve(self):  # for debug
            x = np.arange(0, 255, 1)
            curve = self.cs(x)
            curve[curve > 255] = 255
            plt.figure()
            plt.plot(x, curve)
            plt.plot(x, x)
            plt.plot(self.points, self.values, 'x')
        # plot_ps_curve()
        return output_img
    def exposure_adj(self, img): # to adjust the exposure of a image. The goal is to adjusting the median of value of its gray image to 128
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_count = gray.ravel() # counting the number of each gray value
        self.gray_count = gray_count
        self.gray_median = np.median(gray_count) # find the median value of the gray image
        if np.abs(self.gray_median - 128) > 15:  # adjust the exposure only when the median far away from 128
            adj_fac = np.min([128/self.gray_median, 1.5])  # the factor for adjusting is limited to 1.5 times
            img_out = img.astype(np.float32)
            img_out[img_out < self.gray_median] = img_out[img_out < self.gray_median] * adj_fac # adjusting the portion, which its value smaller than 128
            img_out[img_out >= self.gray_median] = ((img_out[img_out >= self.gray_median] - self.gray_median) / (255 - self.gray_median) * (1 - adj_fac) + adj_fac) * img_out[img_out >= self.gray_median] # adjusting the portion, which its value bigger than 128
            img_out = img_out.astype(np.uint8)
            return img_out
        else:
            return img

    def img_PrePrc(self): # image processing, highlight the lane line pixels and outputs the warped image
        self.original_undst_img = self.cor_dist(self.image) # original image distortion correction
        self.exposure_adjusted = self.exposure_adj(self.image) # adjust the exposure
        self.gamma_adjusted = self.ps_curve_proc(self.exposure_adjusted) # adjust the shallow and brightness of the image
        self.undst_img = self.cor_dist(self.gamma_adjusted)# adjusted image distortion correction

        self.bright_zone = (self.color_thresh(self.original_undst_img[:, :, 0], (120, 255))) & (self.color_thresh(self.original_undst_img[:, :, 1], (120, 255))) # select the bright area in the image. Black lines will be filtered

        # shrinking of bright zone to avoid detecting the line, which divides shallow and bright, as lane line
        self.bright_zone_be = 255 * np.int8(np.invert(self.bright_zone))  # inverse bright and shallow
        self.bright_zone_be = cv2.GaussianBlur(self.bright_zone_be, (9, 9), 0) # the bright area (value > 0) which represents the shallow area in the original image, will be extended
        self.bright_zone_be = self.color_thresh(self.bright_zone_be, thresh=(150, 255), mode=1)  # select how much will the bright area be shrinked
        self.bright_zone_be = np.invert(self.bright_zone_be)

        self.hsv = cv2.cvtColor(self.cor_dist(self.image), cv2.COLOR_RGB2HSV) # turn the color space into HSV
        self.hsv_s = self.hsv[:, :, 1] # select the S-channel
        self.hsv_s_thresh_y = self.color_thresh(self.hsv_s, thresh=(50, 255), mode=1) # select the yellow lane lines. For yellow lane lines usually have high saturation.

        gray_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) # turn the image into gray
        self.gray_img = self.ps_curve_proc(gray_img, mode='gray') # adjust the shallow and brightness areas of the image to filtering the extrem dark area (value < 40) and increasing the contrast

        self.grd = self.abs_sobel_thresh(self.gray_img, orient='x', thresh=(10,80)) # detecting the margin of lane lines

        self.rgb_w = (self.undst_img > [[[190,190,190]]]).all(2) # selecting the white lane lines, for they usually have high values in all three channels
        self.sel = (self.rgb_w | self.hsv_s_thresh_y) & self.bright_zone_be & self.grd  # the detected "yellow" and "white" lines are lane lines, only when they are in bright areas while also coresponding margin are detected. the margins are then as detected lane lines for further use.

        self.sum_blur = cv2.GaussianBlur(255 * np.int8(self.sel), (9, 9), 0)  # denoising on the detected lane lines for finding the lane line bases
        self.sum_compute = 255 * np.int8(self.sel) # for finding the lane line pixels
        ### teste end #####

        self.warped = cv2.warpPerspective(self.sum_blur, M, img_size, flags=cv2.INTER_LINEAR)  # perspective transform
        self.warped_compute = cv2.warpPerspective(self.sum_compute, M, img_size, flags=cv2.INTER_LINEAR)
        self.warped_compute = cv2.GaussianBlur(self.warped_compute, (15, 15), 0)
        self.warped_compute[self.warped_compute < 50] = 0  # denoising the detected lane lins

        self.histogram = np.sum(self.warped[self.warped.shape[0] // 2:, :], axis=0)  # computing the histogram
        midpoint = np.int(self.histogram.shape[0] // 2)
        self.left_base = np.argmax(self.histogram[:midpoint]) # finding the bases of the left and right lane lines
        self.right_base = np.argmax(self.histogram[midpoint:]) + midpoint
        if (self.right_base - self.left_base) * self.xm_per_pix < 2 or (
                self.right_base - self.left_base) * self.xm_per_pix > 5: # preliminarily consider if the detected lane lines are useful or not
            self.frame_drop = True  # this frame will be dropped from detecting lane lines
            # pass
        else:
            self.frame_drop = False
    def find_lane_pixel(self):  # finding lane pixels using sliding window from lane line bases
        nwindows = 9
        margin = 80
        minpix = 100

        window_height = np.int(self.warped_compute.shape[0] // nwindows)

        nonzero = self.warped_compute.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        left_loc = self.left_base
        right_loc = self.right_base

        left_lane_idx = []
        right_lane_idx = []

        for window in range(nwindows):
            win_y_low = self.warped_compute.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_compute.shape[0] - window * window_height
            win_left_low = left_loc - margin
            win_left_high = left_loc + margin
            win_right_low = right_loc - margin
            win_right_high = right_loc + margin

            # cv2.rectangle(warped_compute, (win_left_low, win_y_low), (win_left_high, win_y_high), (255), 2)
            # cv2.rectangle(warped_compute, (win_right_low, win_y_low), (win_right_high, win_y_high), (255), 2)

            good_left_idx = ((nonzeroy > win_y_low) & (nonzeroy < win_y_high) & (nonzerox > win_left_low) & (
                        nonzerox < win_left_high)).nonzero()[0]
            good_right_idx = ((nonzeroy > win_y_low) & (nonzeroy < win_y_high) & (nonzerox > win_right_low) & (
                        nonzerox < win_right_high)).nonzero()[0]

            left_lane_idx.append(good_left_idx)
            right_lane_idx.append(good_right_idx)

            if len(good_left_idx) > minpix:
                left_loc = np.int(np.mean(nonzerox[good_left_idx]))
            if len(good_right_idx) > minpix:
                right_loc = np.int(np.mean(nonzerox[good_right_idx]))

        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)

        self.leftx = nonzerox[left_lane_idx]
        self.lefty = nonzeroy[left_lane_idx]
        self.rightx = nonzerox[right_lane_idx]
        self.righty = nonzeroy[right_lane_idx]

    def draw_poly_eq(self, fac, x): # draw a second order polynom by given factors and x position
        y = fac[0] * x ** 2 + fac[1] * x + fac[2]
        return y
    def fit_poly(self): # finding lane pixels from lane bases
        self.find_lane_pixel()
        if self.lefty.shape[0] < 3 or self.righty.shape[0] < 3: # if the detected lane pixel less than three, then drop this frame from detecting lane lines.
            self.frame_drop = True
            pass
        else:
            self.frame_drop = False
            self.left_fit = np.polyfit(self.lefty, self.leftx, 2) # fit the pixels as lane line using second order polynom
            self.right_fit = np.polyfit(self.righty, self.rightx, 2)

            # using the fitted polynom for the normalized lane line with fixed y-position. This will be used for filtering between frames
            self.ploty = np.linspace(0, self.warped_compute.shape[0] - 1, self.warped_compute.shape[0])
            self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]


            if np.abs(np.max((self.right_fitx - self.left_fitx) * self.xm_per_pix) - np.min((self.right_fitx - self.left_fitx) * self.xm_per_pix)) > 4 or (self.right_fitx < self.left_fitx).any():  # judging if the detected lane lines are valid: 1. if the widths between the widest and the narrowest lane lines greater than 4m, then drop this frame. 2. if the detected left lane line on the right of the detected right lane line, then drop this frame.
                self.frame_drop = True
                pass
            else:
                self.lane_line_filter()  # use this function for smooth detection of lane lines
                self.left_fit = np.polyfit(self.lefty, self.leftx, 2)  # fitting second order polynom after filtering of the lane line
                self.right_fit = np.polyfit(self.righty, self.rightx, 2)

                # for debug
                self.output_img = np.dstack((self.warped_compute, self.warped_compute, self.warped_compute))
                self.output_img[self.lefty, self.leftx] = [255, 255, 0]
                self.output_img[self.righty, self.rightx] = [255, 255, 0]

                self.method = 'fit' # flag for further using

    def search_around_poly(self): # detecting lane lines based on the found validated lane lines
        margin = 50
        nonzero = self.warped_compute.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        self.left_region_l = self.draw_poly_eq(self.left_fit, nonzeroy) - margin # left margin of left lane line from last frame
        self.left_region_r = self.draw_poly_eq(self.left_fit, nonzeroy) + margin
        self.right_region_l = self.draw_poly_eq(self.right_fit, nonzeroy) - margin
        self.right_region_r = self.draw_poly_eq(self.right_fit, nonzeroy) + margin

        left_lane_idx = ((nonzerox > self.left_region_l) & (nonzerox < self.left_region_r)).nonzero()[0]  # pixel index that in the left lane line region
        right_lane_idx = ((nonzerox > self.right_region_l) & (nonzerox < self.right_region_r)).nonzero()[0]

        if (left_lane_idx is None) or (right_lane_idx is None):  # if nothing found
            self.fit_poly()  # then back to the method, that search the lane line from lane base
        else:
            self.frame_drop = False
            self.leftx = nonzerox[left_lane_idx]
            self.lefty = nonzeroy[left_lane_idx]
            self.rightx = nonzerox[right_lane_idx]
            self.righty = nonzeroy[right_lane_idx]

            self.left_fit = np.polyfit(self.lefty, self.leftx, 2)  # fit the pixels as lane line using second order polynom
            self.right_fit = np.polyfit(self.righty, self.rightx, 2)

            # using the fitted polynom for the normalized lane line with fixed y-position. This will be used for filtering between frames
            self.ploty = np.linspace(0, self.warped_compute.shape[0]-1, self.warped_compute.shape[0])
            self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

            if np.abs(np.max((self.right_fitx - self.left_fitx) * self.xm_per_pix) - np.min((self.right_fitx - self.left_fitx) * self.xm_per_pix)) > 4 or (self.right_fitx < self.left_fitx).any(): # judging if the detected lane lines are valid: 1. if the widths between the widest and the narrowest lane lines greater than 4m, then drop this frame. 2. if the detected left lane line on the right of the detected right lane line, then drop this frame.
                self.frame_drop = True
                pass
            else:

                self.lane_line_filter()  # use this function for smooth detection of lane lines
                self.left_fit = np.polyfit(self.lefty, self.leftx, 2)  # fitting second order polynom after filtering of the lane line
                self.right_fit = np.polyfit(self.righty, self.rightx, 2)

                # for debug
                self.output_img = np.dstack((self.warped_compute, self.warped_compute, self.warped_compute))
                self.output_img[self.lefty, self.leftx] = [255, 255, 0]
                self.output_img[self.righty, self.rightx] = [255, 255, 0]

                self.method = 'search_around'# flag for further using

    def measure_curvature(self): # compute the radius of left and right lane lines
        y_eval = np.array([np.max(self.ploty), np.int(np.max(self.ploty) / 2), 0]) # three y-positions: top, middle and bottom
        left_curverad = ((1 + (2 * self.xm_per_pix / self.ym_per_pix ** 2 * self.left_fit[0] * y_eval + self.left_fit[
            1] * self.xm_per_pix / self.ym_per_pix) ** 2) ** 1.5) / np.absolute(2 * self.left_fit[0] * self.xm_per_pix / self.ym_per_pix ** 2)
        right_curverad = ((1 + (2 * self.xm_per_pix / self.ym_per_pix ** 2 * self.right_fit[0] * y_eval + self.right_fit[
            1] * self.xm_per_pix / self.ym_per_pix) ** 2) ** 1.5) / np.absolute(2 * self.right_fit[0] * self.xm_per_pix / self.ym_per_pix ** 2)

        left_base_point = self.left_fit[0] * y_eval ** 2 + self.left_fit[1] * y_eval + self.left_fit[2]  # for computing the lane width of the top, middle and bottom of the lane area
        right_base_point = self.right_fit[0] * y_eval ** 2 + self.right_fit[1] * y_eval + self.right_fit[2]
        lane_width = (right_base_point - left_base_point) * self.xm_per_pix

        offset = ((right_base_point[0] + left_base_point[0]) / 2 - 640) * self.xm_per_pix  # offset of lane middle to vehicle middle

        self.left_curverad = left_curverad.mean()  # average of top, middle and bottom radius of the lane line
        self.right_curverad = right_curverad.mean()
        self.lane_width = lane_width
        self.vehicle_offset = offset

    def lane_line_filter(self):  # filtering the lane lines between frames using normalized lane points
        tiFil_t = [[10, 20, 50, 100, 120, 200], [0.04, 0.05, 0.08, 0.1, 0.8, 1]]  # tiFil_t[0]: difference between two frames in pixel, tiFil_t[1]: time constant for a low pass filter
        if (self.counter <= 1) or self.left_fitx_mem == []: # initialize
            self.left_fitx_mem = self.left_fitx.copy()
            self.right_fitx_mem = self.right_fitx.copy()
        else:
            delta_left = np.abs(self.left_fitx_mem - self.left_fitx)  # compute the difference between two frames of each normalized lane points
            delta_left_max = delta_left.max()
            tiFil_left = self.getZValueFromTabel(delta_left_max, tiFil_t) # using the biggest difference to get the time constant

            delta_right = np.abs(self.right_fitx_mem - self.right_fitx)
            delta_right_max = delta_right.max()
            tiFil_right = self.getZValueFromTabel(delta_right_max, tiFil_t)

            self.left_fitx_mem = self.left_fitx_mem + 0.02 / tiFil_left * (self.left_fitx - self.left_fitx_mem) # compute the x-position of filtered lane points
            self.right_fitx_mem = self.right_fitx_mem + 0.02 / tiFil_right * (self.right_fitx - self.right_fitx_mem)

        self.left_fitx = self.left_fitx_mem.copy()
        self.right_fitx = self.right_fitx_mem.copy()

    def pipe_line(self, img): # processing the current frame for lane line detection
        self.image = img
        self.counter += 1 # the number of current frame
        try:
            self.img_PrePrc()  # image pre-processing to get the warped lane line pixels and lane line base
            if ((self.left_fit is None) and (self.right_fit is None)) or (self.counter == 1 or (np.var(self.lane_width) > 1)) or self.frame_drop == True:
                self.fit_poly()  # fitting the lane line using sliding window, if the previous detection isn't good
            else:
                self.search_around_poly()  # fitting the lane line based on the previous frame, if lane lines of last frame are judged been detected

            self.measure_curvature()  # compute the lane radius and lane width

            # computing the mean value of lane width from past frames and compare it with the lane width of current lane width for judging if the current detected lane line should be dropped.
            # rationale: the lane width of each frame changes very little
            if self.counter <= 1:
                self.lane_width_mean = self.lane_width.mean()
            else:
                if (np.abs(self.lane_width_mean - self.lane_width.mean()) > 1) or (np.var(self.lane_width) > 1) or self.frame_drop == True:
                    self.frame_drop = True
                else:
                    self.frame_drop = False
                self.lane_width_mean = (self.lane_width_mean * (self.counter - 1) + self.lane_width.mean()) / self.counter


            if self.frame_drop == True and self.method == 'search_around': # try sliding window if this frame is dropped and the method of lane line fitting is using self.search_around_poly()
                self.fit_poly()
                self.measure_curvature()

            # post-processing
            if self.frame_drop == True and self.counter > 1: # for dropped frame, the last not dropped frame (self.save.xxx) will be used instead

                warped_zero = np.zeros_like(self.save.warped_compute)
                middle_line = np.zeros_like(self.save.warped_compute)  # for plotting the vehicle middle and the lane middle

                middle_line[:, 638:642] = 255 # vehicle middle

                middle = np.int16((self.save.left_fitx + self.save.right_fitx) / 2)  # lane middle
                lane_middle_x = np.hstack([middle - 2, middle - 1, middle, middle + 1, middle + 2])
                lane_middel_y = np.int16(np.tile(self.save.ploty, (5,)))
                lane_middle = np.zeros_like(self.save.warped_compute)
                lane_middle[lane_middel_y, lane_middle_x] = 255
                color_warped = np.uint8(np.dstack((warped_zero, warped_zero, warped_zero)))

                pts_left = np.array([np.transpose(np.vstack([self.save.left_fitx, self.save.ploty]))])  # plotting the detected lane as a colored block
                pts_right = np.array([np.flipud(np.transpose(np.vstack([self.save.right_fitx, self.save.ploty])))])
                pts = np.hstack((pts_left, pts_right))
                cv2.fillPoly(color_warped, np.int_([pts]), (0, 255, 0))
                color_warped[:, :, 0] = middle_line
                color_warped[:, :, 2] = lane_middle

                color_revert = cv2.warpPerspective(color_warped, Minv, img_size) # change the warped image back to normal perspective
                self.result = cv2.addWeighted(img, 1, color_revert, 0.3, 0)

                if self.save.vehicle_offset < 0: # preparing the text pushed on this frame
                    offset_side = 'right'
                else:
                    offset_side = 'left'
                text = 'left_curverad = ' + str(int(self.save.left_curverad)) + ', right_curverad = ' + str(
                    int(self.save.right_curverad)) + 'frame_drop = ' + str(self.frame_drop)
                text1 = 'lane_width = ' + str("%.2f" % self.save.lane_width.mean()) + ', frame = ' + str(
                    self.counter) + ', Offset = ' + offset_side + ' ' + str("%.2f" % np.abs(self.save.vehicle_offset))

                if self.result[0:200, 0:640, :].mean() < 100: # adapting the text color according to the background
                    color = (255, 255, 255)
                else:
                    color = (0, 0, 0)

                cv2.putText(self.result, text, (100, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=color)
                cv2.putText(self.result, text1, (100, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                            color=color)
                return self.result

            else: # if this frame isn't dropped.
                warped_zero = np.zeros_like(self.warped_compute)
                middle_line = np.zeros_like(self.warped_compute)
                middle_line[:, 638:642] = 255
                middle = np.int16((self.left_fitx + self.right_fitx) / 2)
                lane_middle_x = np.hstack([middle - 2, middle - 1, middle, middle + 1, middle + 2])
                lane_middel_y = np.int16(np.tile(self.ploty, (5,)))
                lane_middle =np.zeros_like(self.warped_compute)
                lane_middle[lane_middel_y, lane_middle_x] = 255
                color_warped = np.uint8(np.dstack((warped_zero, warped_zero, warped_zero)))

                pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
                pts = np.hstack((pts_left, pts_right))

                cv2.fillPoly(color_warped, np.int_([pts]), (0, 255, 0))
                color_warped[:, :, 0] = middle_line
                color_warped[:, :, 2] = lane_middle
                color_revert = cv2.warpPerspective(color_warped, Minv, img_size)

                self.result = cv2.addWeighted(img, 1, color_revert, 0.3, 0)

                if self.vehicle_offset < 0:
                    offset_side = 'right'
                else:
                    offset_side = 'left'
                text = 'left_curverad = ' + str(int(self.left_curverad)) + ', right_curverad = ' + str(int(self.right_curverad)) + 'frame_drop = ' + str(self.frame_drop)
                text1 = 'lane_width = ' + str("%.2f" % self.lane_width.mean()) + ', frame = ' + str(self.counter) + ', Offset = ' + offset_side + ' ' + str("%.2f" % np.abs(self.vehicle_offset))
                if self.result[0:200, 0:640, :].mean() < 100:
                    color = (255, 255, 255)
                else:
                    color = (0,0,0)
                cv2.putText(self.result, text, (100, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color = color)
                cv2.putText(self.result, text1, (100, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=color)

                self.save = [] # clear the self.save
                self.save = copy.deepcopy(self) # take the current result to self.save
                return self.result
        except Exception as err:
            print(err)
            plt.imsave('debug/last.jpg', img)
            return None