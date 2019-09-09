import time
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# SmashScan libraries
import util

# https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
# 7 - single text line, 8 - single word, 8 works well with background blobs.

# An object that takes the results from VideoAnalyzer and performs OCR over
# the frames in frame_range to generate a time series of damage percents.
class OCRAnalyzer:

    def __init__(self, capture, match_bboxes=None, gray_flag=True, denoise_flag=True, save_flag=False, show_flag=False, wait_flag=False):
        self.capture = capture
        self.gray_flag = gray_flag
        self.save_flag = save_flag
        self.show_flag = show_flag
        self.match_bboxes = match_bboxes
        self.denoise_flag = denoise_flag
        # cv2.namedWindow('frame_slice')
        # cv2.namedWindow('bin_frame')
        # cv2.namedWindow('bin_alt')
        # cv2.namedWindow('result')

    def standard_test(self, frame_range, ports):
        time_series = list() # Format: [[Frame number, P1 damage, P2 damage, P3 damage, P4 damage] ...
        for fnum in frame_range:
            if self.show_flag: print('FNUM:', fnum)
            frame_data = [fnum]
            frame = util.get_frame(self.capture, fnum, gray_flag=self.gray_flag)
            for port in ports:
                if self.show_flag: print('PORT:', port)
                frame_slice = self.get_frame_roi(frame, port)
                frame_slice_bin = self.filter_percentage(frame_slice)
                damage = self.analyze_frame(frame_slice_bin, '7')
                frame_data.append(damage)
            time_series.append(frame_data)
        # print(time_series)
        time_series = np.array(time_series)
        print(time_series[:10])
        # print('ACCURACY:', len([val for fnum, val in time_series if val != -100])/len(time_series))
        if self.denoise_flag: self.denoise(time_series)
        plt.plot(time_series[:, 0], time_series[:, 1:])
        plt.legend(ports)
        plt.show()

    def analyze_frame(self, frame, flag):
        start_time = time.time()
        text = pytesseract.image_to_string(frame, lang="smash", config='--tessdata-dir ./tessdata/ --psm ' + flag)
        # print(text)
        # util.display_total_time(start_time)
        #
        # start_time = time.time()
        # pytess_result = pytesseract.image_to_boxes(frame, lang="smash", config='--tessdata-dir ./tessdata/', output_type=pytesseract.Output.DICT)
        # print(pytess_result)
        # util.display_total_time(start_time)
        # bbox_list = list()
        # for i, _ in enumerate(pytess_result['bottom']):
        #     tl = (pytess_result['left'][i], pytess_result['bottom'][i])
        #     br = (pytess_result['right'][i], pytess_result['top'][i])
        #     bbox_list.append((tl, br))
        #     print(pytess_result)
        #     print('Text:', pytess_result['char'][i])
        #     util.show_frame(frame, bbox_list=[(tl, br)], wait_flag=True)

        start_time = time.time()
        pytess_data = pytesseract.image_to_data(frame, lang="smash", config='--tessdata-dir ./tessdata/ --psm ' + flag,
                                                output_type=pytesseract.Output.DICT)

        bbox_list = list()
        final_text = ''
        for i, conf in enumerate(pytess_data['conf']):
            if int(conf) != -1:
                text = pytess_data['text'][i]
                if conf > 30:
                    final_text += text
                if self.show_flag:
                    print("\tconf: {}".format(conf))
                    tl = (pytess_data['left'][i], pytess_data['top'][i])
                    br = (tl[0] + pytess_data['width'][i], tl[1] + pytess_data['height'][i])
                    bbox_list.append((tl, br))
                    print('Text:', text)
                    util.show_frame(frame, bbox_list=[(tl, br)], wait_flag=True)

        # Return highest confidence result
        final_text = final_text.replace('%', '')
        final_text = ''.join(final_text.split())
        # if final_text == '':
            # util.show_frame(frame, bbox_list=[], wait_flag=Truej)
        return None if final_text == '' else int(final_text)

    # Perform contour filtering on a binarized percentage slice
    def filter_percentage(self, frame_slice):
        if self.show_flag:
            # cv2.imshow('frame_slice', frame_slice)
            util.show_frame(frame_slice, bbox_list=[], wait_flag=True)
        # if self.show_flag: print('SIZE:', frame_slice.size)
        show_contours = True
        frame_blur = cv2.bilateralFilter(frame_slice, 9, 75, 75)
        if self.show_flag: util.show_frame(frame_blur, bbox_list=[], wait_flag=True)
        _, bin_frame = cv2.threshold(frame_blur, 100, 255, cv2.THRESH_BINARY)
        # bin_frame = cv2.adaptiveThreshold(frame_slice, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, -30)
        _, contours, hierarchy = cv2.findContours(bin_frame, cv2.RETR_CCOMP,
                                                  cv2.CHAIN_APPROX_NONE)

        if self.show_flag: util.show_frame(bin_frame, bbox_list=[], wait_flag=True)# cv2.imshow('bin_frame', bin_frame)

        if len(contours) is 0:
            return np.zeros(bin_frame.shape, np.uint8) # Don't waste time filtering contours on a blank image

        hierarchy = hierarchy[0]  # Why does this list have a 3rd dimension? Let's remove it

        if self.show_flag and show_contours:
            self.show_contours(frame_slice, contours, hierarchy)

        # Run two passes over the detected contours. On the first pass, mark any contour that falls outside the area
        # threshold for deletion. On the second pass, mark any contour that is a child of another marked contour.
        percentage_contours = []
        to_delete = []
        # For edge contact detection
        frame_h = frame_slice.shape[0]
        frame_w = frame_slice.shape[1]
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w/h
            solidity = area / max(0.5, cv2.contourArea(cv2.convexHull(contour)))
            if (hierarchy[i][3] == -1 and
                    (x == 0 or y == 0 or x+w == frame_w or y+h == frame_h# Is contour touching border of the frame?
                     # or area < .005 * frame_slice.size or area > .1 * frame_slice.size or
                     or area < 100 or area > 450
                     or h < 20 or h > 25
                     or w < 10 or w > 23
                     # or area > 300 and area < 450
                     or aspect_ratio < 0.4 or aspect_ratio > 1.0
                     or solidity < 0.45
                    )):

                to_delete.append(i)
        for i, contour in enumerate(contours):
            parent_idx = hierarchy[i][3]
            if parent_idx in to_delete:
                to_delete.append(i)
        # Copy unmarked contours to new array
        for i, contour in enumerate(contours):
            if i not in to_delete:
                percentage_contours.append(contour)

        if self.show_flag and show_contours:
            self.show_contours(frame_blur, percentage_contours, hierarchy)

        bin_alt = np.zeros(bin_frame.shape, np.uint8)
        cv2.drawContours(bin_alt, percentage_contours, -1, 255, cv2.FILLED)
        if self.show_flag:
            util.show_frame(bin_alt, bbox_list=[], wait_flag=True)
            # cv2.imshow('bin_alt', bin_alt)
        # bin_alt = util.resize_img(bin_alt, 4)
        return bin_alt

    # Given a full frame of match footage, return a rectangular region corresponding to the damage percentage of a particular player
    def get_frame_roi(self, frame, port=None):
        # Get size and position of match bounding box relative to the origin of the frame
        match_x = self.match_bboxes[0][0]
        match_y = self.match_bboxes[0][1]
        match_width = self.match_bboxes[1][0] - match_x
        match_height = self.match_bboxes[1][1] - match_y
        # Shrink match bounding box horizontally so P1 and P4 percentages touch the border
        # match_x += int(1/30 * match_width)
        # match_width -= int(1/30 * match_width)
        # Calculate the ROI bounding box
        top = int(10/12 * match_height) + match_y - 10
        bottom = int(113/120 * match_height) + match_y + 10
        left = int(1/4 * (port - 1) * match_width) + match_x
        right = int(left+match_width) if port is None else int(1/4 * port * match_width) + match_x
        return frame[top:, left:right]

    # Debugging function that highlights each contour in a list and provides various characteristics
    def show_contours(self, frame_blur, contours, hierarchy):
        for i, contour in enumerate(contours):
            bin_contour = np.array(frame_blur, np.uint8)
            cv2.drawContours(bin_contour, [contour], 0, 255, 1)
            # Compute and display various contour features to guide our contour filter`
            area = cv2.contourArea(contour)
            print('AREA:', area)
            # Solidity = area / convex hull area
            solidity = area / max(0.5, cv2.contourArea(cv2.convexHull(contour)))
            print('SOLIDITY:', solidity)
            print('PARENT?', hierarchy[i][3] != -1)
            x, y, w, h = cv2.boundingRect(contour)
            print('ARATIO:', w / h)
            print('WIDTH:', w, 'HEIGHT:', h)
            print('LEFT:', x, 'RIGHT:', x + w, 'TOP:', y, 'BOTTOM:', y + h)
            util.show_frame(bin_contour, bbox_list=[], wait_flag=True)
            # cv2.imshow('contour', bin_contour)
            # cv2.waitKey(0)

    # Since the jumble effect on recently modified percentages inevitably throws OCR off, we manually correct suspicious
    # values based on values in the immediate past and future
    def denoise(self, time_series):

        # Inner function that identifies a suspicious region in the specified range and direction of time_series
        def flatten_region(range, target_value, reverse=False):
            if reverse: range = range[::-1]
            for idx in range:
                value = time_series[idx, port]
                # Check if value >= target_value in the forward direction and value <= target_value in the reverse
                if value is None or value == target_value or (value < target_value) == reverse:
                    return idx
            return -1

        fnum, time_series = time_series[:, 0:1], time_series[:, 1:]
        # First pass: Flatten - remove sudden changes in percentage that last a single frame.
        i = 1
        while i < len(time_series)-1: # Ignore first and last frames for now
            percents = time_series[i]
            for port, current_percent in enumerate(percents):
                previous_percent = time_series[i-1, port]
                next_percent = time_series[i+1, port]
                if None not in [previous_percent, current_percent, next_percent]:
                    if current_percent < previous_percent:
                        # print('DEBUG:', time_series[i, port], time_series[i-1, port])
                        # Iterate forward, checking if a value equal or greater than current_percent exists in the next
                        # n frames (where n is a tuned constant representing the maximum number of frames a noisy region
                        # is expected to last
                        # TODO: Compare n to fnum
                        n = 10
                        j = flatten_region(range(i+1, min(i+n+1, time_series.shape[0])), previous_percent)
                        if j != -1:
                            # print('DEBUG:', time_series[i - 1:j + 1, port], np.repeat(previous_percent, j - i))
                            time_series[i:j, port] = np.repeat(previous_percent, j - i)
                            # print('DEBUG:', time_series[i - 1:j + 1, port])
                            # i = j
                        else:
                            j = flatten_region(range(0, i-1), current_percent, reverse=True)
                            if j != -1:
                                # print('DEBUG:', time_series[j:i+1, port], np.repeat(time_series[j, port], i-j-1))
                                time_series[j+1:i, port] = np.repeat(time_series[j, port], i-j-1)
                                # print('DEBUG:', time_series[j:i + 1, port])
            i += 1

                # Suspicious values occur when a damage value is lower than the value immediately preceding it in time,
                # since a decrease in damage should not occur in competitive play. Our objective is to determine which
                # of the larger or smaller values is erroneous.
        time_series = np.hstack((fnum, time_series))

    # Is one number a broken substring of another? (i.e. 20 --> 120 or 18 --> 108) Necessary for denoising
    def is_broken_substring(self, val_1, val_2):
        # The number of digits in the smaller number that correspond to one in the larger number; in a broken substring,
        # all digits correspond, so common_digits =
        common_digits = 0
        # The position in val_2 we start searching from with the current digit. Shifting the start point prevents
        # out-of-order matches, such as 12 == 21.
        start = 0
        val_1, val_2 = str(min(val_1, val_2)), str(max(val_1, val_2))
        for digit in val_1:
            for i in range(start, len(val_2)):
                if val_2[i] == digit:
                    common_digits += 1
                    start = i+1
                    break
        return common_digits == len(val_1)

    def ocr_test(self, img, hsv_flag, avg_flag=False, gau_flag=False,
        med_flag=False, bil_flag=False, inv_flag=True):

        # Create a grayscale and HSV copy of the input image.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # If the HSV flag is enabled, select white OR red -> (High S AND Mid H)'
        if hsv_flag:
            # mask = cv2.inRange(img_hsv, (15, 50, 0), (160, 255, 255))
            mask = cv2.inRange(img_hsv, (15, 50, 0), (160, 255, 255))
            result_img = cv2.bitwise_and(img_gray, img_gray,
                mask=cv2.bitwise_not(mask))
        # else:
            result_img = img_gray

        # Apply a post blurring filter according to the input flag given.
        # https://docs.opencv.org/3.4.5/d4/d13/tutorial_py_filtering.html
        if avg_flag:
            result_img = cv2.blur(result_img, (5, 5))
        elif gau_flag:
            result_img = cv2.GaussianBlur(result_img, (5, 5), 0)
        elif med_flag:
            result_img = cv2.medianBlur(result_img, 5)
        elif bil_flag:
            result_img = cv2.bilateralFilter(result_img, 9, 75, 75)

        # Invert the image to give the image a black on white background.
        if inv_flag:
            result_img = cv2.bitwise_not(result_img)

        display_ocr_test_flags(hsv_flag, avg_flag, gau_flag,
            med_flag, bil_flag, inv_flag)
        show_ocr_result(result_img)


    # Display the OCR test flags in a structured format.
    def display_ocr_test_flags(self, hsv_flag, avg_flag, gau_flag,
        med_flag, bil_flag, inv_flag):
        print("hsv_flag={}".format(hsv_flag))

        if avg_flag:
            print("avg_flag={}".format(avg_flag))
        elif gau_flag:
            print("gau_flag={}".format(gau_flag))
        elif med_flag:
            print("med_flag={}".format(med_flag))
        elif bil_flag:
            print("bil_flag={}".format(bil_flag))

        print("inv_flag={}".format(inv_flag))


    def contour_test(self, img):
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        img_d = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_d, contours, -1, (255, 0, 0), 2)
        cv2.imshow('test', img_d)
        cv2.waitKey(0)
        res = np.zeros(img.shape, np.uint8)

        for i, contour in enumerate(contours):
            img_d = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_d, contour, -1, (255, 0, 0), 3)

            moment = cv2.moments(contour)
            if moment['m00']: # Removes single points
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
                print("Center: {}".format((cx, cy)))
                cv2.circle(img_d, (cx, cy), 3, (0, 0, 255), -1)

            print("Area: {}".format(cv2.contourArea(contour)))
            print("Permeter: {} ".format(cv2.arcLength(contour, True)))

            cv2.imshow('test', img_d)
            cv2.waitKey(0)

            # The result displayed is an accumulation of previous contours.
            mask = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
            mask = cv2.bitwise_and(img, mask)
            res = cv2.bitwise_or(res, mask)
            cv2.imshow('test', res)
            cv2.waitKey(0)


if __name__ == '__main__':
    capture = cv2.VideoCapture("videos/doubles.mp4")
    # ocr = OCRAnalyzer(capture, show_flag=True, match_bboxes=((0, 0), (884, 706)))
    ocr = OCRAnalyzer(capture, show_flag=False, match_bboxes=((0, 0), (442, 353)), denoise_flag=False)
    # ocr.standard_test(frame_range=range(510, 3 * 60 * 30 + 27 * 30), ports=[2])
    # ocr.standard_test(frame_range=range(510, 1 * 60 * 30 + 15 * 30), ports=[2])
    # ocr.standard_test(frame_range=range(510, 1000), ports=[1, 2, 3, 4])

    # for fnum in [120, 9000]: # 3400 works fine
    #     capture = cv2.VideoCapture("videos/test.mp4")
    #     frame = util.get_frame(capture, fnum, gray_flag=True)
    #     frame = frame[300:340, 80:220] # 300:340, 200:320
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey(0)

    #     #frame = cv2.imread('videos/test4.png', cv2.IMREAD_GRAYSCALE)
    #     #show_ocr_result(frame)

    #     #img2 = cv2.imread('videos/test4.png', cv2.IMREAD_COLOR)
    #     #ocr_test(img2, hsv_flag=False)
    #     #ocr_test(img2, hsv_flag=False, avg_flag=True)
    #     #ocr_test(img2, hsv_flag=False, gau_flag=True)
    #     #ocr_test(img2, hsv_flag=False, med_flag=True)
    #     #ocr_test(img2, hsv_flag=False, bil_flag=True)

    #     #ocr_test(img2, hsv_flag=True)
    #     #ocr_test(img2, hsv_flag=True, avg_flag=True)
    #     #ocr_test(img2, hsv_flag=True, gau_flag=True)
    #     #ocr_test(img2, hsv_flag=True, med_flag=True)
    #     #ocr_test(img2, hsv_flag=True, bil_flag=True)

    #     # https://docs.opencv.org/3.4.5/d7/d4d/tutorial_py_thresholding.html
    #     print("thresh")
    #     blur = cv2.GaussianBlur(frame, (5, 5), 0)
    #     _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    #     th = cv2.medianBlur(thresh, 5)
    #     show_ocr_result(cv2.bitwise_not(th))

    #     print("adaothresh")
    #     _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    #     show_ocr_result(cv2.bitwise_not(th2))

    #     contour_test(th2)
