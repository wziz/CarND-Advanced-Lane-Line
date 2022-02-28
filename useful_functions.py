#import functions as fcn
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import glob
import math
from collections import Counter
from scipy.interpolate import CubicSpline
import copy
import operator

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


class frame_class(object):
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
        points = np.array([0, 20, 55, 90, 160, 200, 230, 255])
        values = np.array([0, 10, 50, 90, 160, 220, 250, 255])
        self.cs = CubicSpline(points, values)
        points_gray = np.array([0,40,255])
        values_gray = np.array([0, 0, 255])
        self.cs_gray = CubicSpline(points_gray, values_gray)
        self.method = ''
    #  useful functions
    def grayshow(self, image, t=None):
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(t)
    def output_mode(self, select_region, mode):
        if mode == 1:
            output = select_region
        elif mode == 2 or mode == 3:
            binary_output = np.zeros_like(select_region)
            binary_output[select_region] = 1
            if mode == 2:
                output = binary_output
                plt.figure()
                plt.imshow(binary_output, cmap='gray')
            else:
                output = [binary_output, select_region]
        return output
    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0,255), mode=1):  # image here should have only one channel
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
    def mag_thresh(self, image, sobel_kernel= 3, mag_thresh=(0,255), mode=1):
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
    def dir_thresh(self, image, sobel_kernel=3, thresh=(0, np.pi/2), mode=1):
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
    def color_thresh(self, image, thresh=(0,255), mode=1):
        select_region = ((image >= thresh[0]) & (image <= thresh[1]))
        output = self.output_mode(select_region, mode)
        return output
    def cor_dist(self, image, mtx=mtx_calc, dist=dist_calc):
        undst = cv2.undistort(image, mtx, dist, None, mtx)
        return undst
    def getZValueFromTabel(self, input_x, tiFil_tabel):
        i = 0
        while input_x > tiFil_tabel[0][i] and i < len(tiFil_tabel[0]) - 1:
            i += 1
        return tiFil_tabel[1][i]
    def contrast_adj(self, image, contrast):
        buf = image.copy()
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        return buf
    def gamma_curve_proc(self, input_img, mode='color'):
        # def plot_gamma_curve():
        #     x = np.arange(0, 255, 1)
        #     curve = self.cs(x)
        #     curve[curve > 255] = 255
        #     plt.figure()
        #     plt.plot(x, curve)
        #     plt.plot(x, x)
        #     plt.plot(points, values, 'x')
        if mode == 'color':
            cs = self.cs
        elif mode == 'gray':
            cs = self.cs_gray
        else:
            cs = self.cs
        output_img = input_img.copy()
        output_img = cs(output_img)
        output_img[output_img > 255] = 255
        output_img = output_img.astype(np.uint8)
        return output_img
        # plot_gamma_curve()
        # if len(input_img.shape) > 2:
        #     gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        # else:
        #     gray = input_img.copy()
        # output_img = input_img.copy()
        # for i in range(1, points.shape[0]):
        #     if i == 1:
        #         thresh_start = 0
        #     else:
        #         thresh_start = points[i-1]
        #     thresh_end = points[i]
        #     sel_zone = self.color_thresh(gray, thresh=(thresh_start, thresh_end), mode=1)
        #     output_img[sel_zone.nonzero()[0], sel_zone.nonzero()[1]]
    def exposure_adj(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_count = gray.ravel() #gray[gray.shape[0] // 2:, :].ravel()
        self.gray_count = gray_count
        self.gray_median = np.median(gray_count)
        if np.abs(self.gray_median - 128) > 15:
            adj_fac = np.min([128/self.gray_median, 1.5])
            img_out = img.astype(np.float32)
            img_out[img_out < self.gray_median] = img_out[img_out < self.gray_median] * adj_fac
            img_out[img_out >= self.gray_median] = ((img_out[img_out >= self.gray_median] - self.gray_median) / (255 - self.gray_median) * (1 - adj_fac) + adj_fac) * img_out[img_out >= self.gray_median]
            img_out = img_out.astype(np.uint8)
            return img_out
        else:
            return img
    #  image processing
    def img_PrePrc(self):
        self.original_undst_img = self.cor_dist(self.image)
        self.exposure_adjusted = self.exposure_adj(self.image)
        self.gamma_adjusted = self.gamma_curve_proc(self.exposure_adjusted)
        self.undst_img = self.cor_dist(self.gamma_adjusted)# undistort image

        self.bright_zone = (self.color_thresh(self.original_undst_img[:, :, 0], (120, 255))) & (self.color_thresh(self.original_undst_img[:, :, 1], (120, 255)))
        # shrank of bright zone
        self.bright_zone_be = 255 * np.int8(np.invert(self.bright_zone))
        self.bright_zone_be = cv2.GaussianBlur(self.bright_zone_be, (9, 9), 0)
        self.bright_zone_be = self.color_thresh(self.bright_zone_be, thresh=(150, 255), mode=1)
        self.bright_zone_be = np.invert(self.bright_zone_be)

        self.hsv = cv2.cvtColor(self.cor_dist(self.image), cv2.COLOR_RGB2HSV)
        self.hsv_s = self.hsv[:, :, 1]
        self.hsv_s_thresh_y = self.color_thresh(self.hsv_s, thresh=(50, 255), mode=1)

        gray_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.gray_img = self.gamma_curve_proc(gray_img, mode='gray')

        self.grd = self.abs_sobel_thresh(self.gray_img, orient='x', thresh=(10,80))

        self.rgb_w = (self.undst_img > [[[190,190,190]]]).all(2)
        self.sel = (self.rgb_w | self.hsv_s_thresh_y) & self.bright_zone_be & self.grd

        self.sum_blur = cv2.GaussianBlur(255 * np.int8(self.sel), (9, 9), 0)
        self.sum_compute = 255 * np.int8(self.sel)
        ### teste end #####

        self.warped = cv2.warpPerspective(self.sum_blur, M, img_size, flags=cv2.INTER_LINEAR)  # perspective transform
        self.warped_compute = cv2.warpPerspective(self.sum_compute, M, img_size, flags=cv2.INTER_LINEAR)
        self.warped_compute = cv2.GaussianBlur(self.warped_compute, (15, 15), 0)
        self.warped_compute[self.warped_compute < 50] = 0

        self.histogram = np.sum(self.warped[self.warped.shape[0] // 2:, :], axis=0)  # computing the histogram
        midpoint = np.int(self.histogram.shape[0] // 2)
        self.left_base = np.argmax(self.histogram[:midpoint])
        self.right_base = np.argmax(self.histogram[midpoint:]) + midpoint
        if (self.right_base - self.left_base) * self.xm_per_pix < 2 or (
                self.right_base - self.left_base) * self.xm_per_pix > 5:
            self.frame_drop = True
            # pass
        else:
            self.frame_drop = False
    def find_lane_pixel(self):
        nwindows = 9
        margin = 80
        minpix = 100
        # to do: search should begin at the lowest position, where there is nonzero point. so the window_height shou
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
        # self.valid_founded_lane_pixel()
    def draw_poly_eq(self, fac, x):
        y = fac[0] * x ** 2 + fac[1] * x + fac[2]
        return y
    def fit_poly(self):
        self.find_lane_pixel()
        if self.lefty.shape[0] < 3 or self.righty.shape[0] < 3:
            self.frame_drop = True
            pass
        else:
            self.frame_drop = False
            self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
            self.right_fit = np.polyfit(self.righty, self.rightx, 2)

            self.ploty = np.linspace(0, self.warped_compute.shape[0] - 1, self.warped_compute.shape[0])
            self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

            if np.abs(np.max((self.right_fitx - self.left_fitx) * self.xm_per_pix) - np.min((self.right_fitx - self.left_fitx) * self.xm_per_pix)) > 4 or (self.right_fitx < self.left_fitx).any():
                self.frame_drop = True
                pass
            else:
                self.lane_line_filter()
                self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
                self.right_fit = np.polyfit(self.righty, self.rightx, 2)

                self.output_img = np.dstack((self.warped_compute, self.warped_compute, self.warped_compute))
                self.output_img[self.lefty, self.leftx] = [255, 255, 0]
                self.output_img[self.righty, self.rightx] = [255, 255, 0]
                self.method = 'fit'
    def search_around_poly(self):
        margin = 50
        nonzero = self.warped_compute.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        self.left_region_l = self.draw_poly_eq(self.left_fit, nonzeroy) - margin # left_fitx from last frame
        self.left_region_r = self.draw_poly_eq(self.left_fit, nonzeroy) + margin
        self.right_region_l = self.draw_poly_eq(self.right_fit, nonzeroy) - margin
        self.right_region_r = self.draw_poly_eq(self.right_fit, nonzeroy) + margin

        left_lane_idx = ((nonzerox > self.left_region_l) & (nonzerox < self.left_region_r)).nonzero()[0]
        right_lane_idx = ((nonzerox > self.right_region_l) & (nonzerox < self.right_region_r)).nonzero()[0]

        if (left_lane_idx is None) or (right_lane_idx is None):
            self.fit_poly()
        else:
            self.frame_drop = False
            self.leftx = nonzerox[left_lane_idx]
            self.lefty = nonzeroy[left_lane_idx]
            self.rightx = nonzerox[right_lane_idx]
            self.righty = nonzeroy[right_lane_idx]

            # self.valid_founded_lane_pixel()

            self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
            self.right_fit = np.polyfit(self.righty, self.rightx, 2)
            self.ploty = np.linspace(0, self.warped_compute.shape[0]-1, self.warped_compute.shape[0])
            self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

            if np.abs(np.max((self.right_fitx - self.left_fitx) * self.xm_per_pix) - np.min((self.right_fitx - self.left_fitx) * self.xm_per_pix)) > 4 or (self.right_fitx < self.left_fitx).any():
                self.frame_drop = True
                pass
            else:

                self.lane_line_filter()
                self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
                self.right_fit = np.polyfit(self.righty, self.rightx, 2)
                self.method = 'search_around'
    def measure_curvature(self):
        y_eval = np.array([np.max(self.ploty), np.int(np.max(self.ploty) / 2), 0])
        left_curverad = ((1 + (2 * self.xm_per_pix / self.ym_per_pix ** 2 * self.left_fit[0] * y_eval + self.left_fit[
            1] * self.xm_per_pix / self.ym_per_pix) ** 2) ** 1.5) / np.absolute(2 * self.left_fit[0] * self.xm_per_pix / self.ym_per_pix ** 2)
        right_curverad = ((1 + (2 * self.xm_per_pix / self.ym_per_pix ** 2 * self.right_fit[0] * y_eval + self.right_fit[
            1] * self.xm_per_pix / self.ym_per_pix) ** 2) ** 1.5) / np.absolute(2 * self.right_fit[0] * self.xm_per_pix / self.ym_per_pix ** 2)
        left_base_point = self.left_fit[0] * y_eval ** 2 + self.left_fit[1] * y_eval + self.left_fit[2]
        right_base_point = self.right_fit[0] * y_eval ** 2 + self.right_fit[1] * y_eval + self.right_fit[2]
        lane_width = (right_base_point - left_base_point) * self.xm_per_pix
        offset = ((right_base_point[0] + left_base_point[0]) / 2 - 640) * self.xm_per_pix
        # if lane_width.any() > 4.3 or lane_width.any() < 3.5:
        #     # pass
        #     self.frame_drop = True
        # else:
        #     self.frame_drop = False
        self.left_curverad = left_curverad.mean()
        self.right_curverad = right_curverad.mean()
        self.lane_width = lane_width
        self.vehicle_offset = offset
    def lane_line_filter(self):
        tiFil_t = [[10, 20, 50, 100, 120, 200], [0.04, 0.05, 0.08, 0.1, 0.8, 1]]
        if (self.counter <= 1) or self.left_fitx_mem == []:
            self.left_fitx_mem = self.left_fitx.copy()
            self.right_fitx_mem = self.right_fitx.copy()
        else:
            delta_left = np.abs(self.left_fitx_mem - self.left_fitx)
            delta_left_max = delta_left.max()
            tiFil_left = self.getZValueFromTabel(delta_left_max, tiFil_t)

            delta_right = np.abs(self.right_fitx_mem - self.right_fitx)
            delta_right_max = delta_right.max()
            tiFil_right = self.getZValueFromTabel(delta_right_max, tiFil_t)

            self.left_fitx_mem = self.left_fitx_mem + 0.02 / tiFil_left * (self.left_fitx - self.left_fitx_mem)
            self.right_fitx_mem = self.right_fitx_mem + 0.02 / tiFil_right * (self.right_fitx - self.right_fitx_mem)

        self.left_fitx = self.left_fitx_mem.copy()
        self.right_fitx = self.right_fitx_mem.copy()
    def pipe_line(self, img):
        self.image = img
        self.counter += 1
        try:
            self.img_PrePrc()  # warped, left_base, right_base, warped_compute, histogram
            if ((self.left_fit is None) and (self.right_fit is None)) or (self.counter == 1 or (np.var(self.lane_width) > 1)) or self.frame_drop == True:
                self.fit_poly()  # ploty, left_fitx, right_fitx, output_img
            else:
                self.search_around_poly()  # ploty, left_fitx, right_fitx

            self.measure_curvature()  # left_curverad, right_curverad, lane_width, vehicle_offset

            if self.counter <= 1:
                self.lane_width_mean = self.lane_width.mean()
            else:
                if (np.abs(self.lane_width_mean - self.lane_width.mean()) > 1) or (np.var(self.lane_width) > 1) or self.frame_drop == True:
                    self.frame_drop = True
                else:
                    self.frame_drop = False
                self.lane_width_mean = (self.lane_width_mean * (self.counter - 1) + self.lane_width.mean()) / self.counter

            if self.frame_drop == True and self.method == 'search_around':
                self.fit_poly()
                self.measure_curvature()

            if self.frame_drop == True and self.counter > 1:
                warped_zero = np.zeros_like(self.save.warped_compute)
                middle_line = np.zeros_like(self.save.warped_compute)
                middle_line[:, 638:642] = 255
                middle = np.int16((self.save.left_fitx + self.save.right_fitx) / 2)
                lane_middle_x = np.hstack([middle - 2, middle - 1, middle, middle + 1, middle + 2])
                lane_middel_y = np.int16(np.tile(self.save.ploty, (5,)))
                lane_middle = np.zeros_like(self.save.warped_compute)
                lane_middle[lane_middel_y, lane_middle_x] = 255
                color_warped = np.uint8(np.dstack((warped_zero, warped_zero, warped_zero)))

                pts_left = np.array([np.transpose(np.vstack([self.save.left_fitx, self.save.ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([self.save.right_fitx, self.save.ploty])))])
                pts = np.hstack((pts_left, pts_right))

                cv2.fillPoly(color_warped, np.int_([pts]), (0, 255, 0))
                color_warped[:, :, 0] = middle_line
                color_warped[:, :, 2] = lane_middle
                color_revert = cv2.warpPerspective(color_warped, Minv, img_size)

                self.result = cv2.addWeighted(img, 1, color_revert, 0.3, 0)

                if self.save.vehicle_offset < 0:
                    offset_side = 'right'
                else:
                    offset_side = 'left'
                text = 'left_curverad = ' + str(int(self.save.left_curverad)) + ', right_curverad = ' + str(
                    int(self.save.right_curverad)) + 'frame_drop = ' + str(self.frame_drop)
                text1 = 'lane_width = ' + str("%.2f" % self.save.lane_width.mean()) + ', frame = ' + str(
                    self.counter) + ', Offset = ' + offset_side + ' ' + str("%.2f" % np.abs(self.save.vehicle_offset))
                if self.result[0:200, 0:640, :].mean() < 100:
                    color = (255, 255, 255)
                else:
                    color = (0, 0, 0)
                cv2.putText(self.result, text, (100, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=color)
                cv2.putText(self.result, text1, (100, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                            color=color)
                return self.result
            else:
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
                self.save = []
                self.save = copy.deepcopy(self)
                return self.result
        except Exception as err:
            print(err)
            plt.imsave('debug/last.jpg', img)
            return None