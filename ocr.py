import time
import cv2
import pytesseract
import numpy as np

# SmashScan libraries
import util

# https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
# 7 - single text line, 8 - single word, 8 works well with background blobs.

def show_ocr_result(frame, flag):
    # start_time = time.time()
    # text = pytesseract.image_to_string(frame, config="--psm " + flag)
    # print(text)
    # util.display_total_time(start_time)

    # start_time = time.time()
    # pytess_result = pytesseract.image_to_boxes(frame,
    #     config="--psm " + flag, output_type=pytesseract.Output.DICT)
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
    pytess_data = pytesseract.image_to_data(frame, 
        config="--psm " + flag, output_type=pytesseract.Output.DICT)
    # print(pytess_data)
    util.display_total_time(start_time)

    bbox_list = list()
    for i, conf in enumerate(pytess_data['conf']):
        if int(conf) != -1:
            print("\tconf: {}".format(conf))
            tl = (pytess_data['left'][i], pytess_data['top'][i])
            br = (tl[0]+pytess_data['width'][i], tl[1]+pytess_data['height'][i])
            bbox_list.append((tl, br))
            print('Text:', pytess_data['text'][i])
            util.show_frame(frame, bbox_list=[(tl, br)], wait_flag=True)


def ocr_test(img, hsv_flag, avg_flag=False, gau_flag=False,
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
    else:
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
def display_ocr_test_flags(hsv_flag, avg_flag, gau_flag,
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


def contour_test(img):
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


capture = cv2.VideoCapture("videos/test_720.mp4")
# for fnum in range(0, 9000):
    # frame = util.get_frame(capture, fnum, gray_flag=True)
    # frame = frame[int(0.75 * frame.shape[0]):, :]
    # _, bin_frame = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('frame', frame)
    # cv2.imshow('binarized', bin_frame)
    # cv2.waitKey(1)

show_images = True
for fnum in range(120, 2400, 120):
    print('FRAME:', fnum)
# for fnum in [1200, 600, 1800, 2400]:
    frame = util.get_frame(capture, fnum, gray_flag=True)
    # frame = frame[int(0.75 * frame.shape[0]):, :]
    frame = frame[600:670, 280:430]
    if show_images:
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
    frame_list = frame.tolist()
    frame_list = [['***' if val < 60 else '%03d' % val for val in row] for row in frame_list]
    # for row in frame_list:
    #     print(row)
    _, bin_frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

    if show_images:
        cv2.imshow('Binarized:', bin_frame)
        cv2.waitKey(0)

   # Separate the percentage from the surrounding white background
    _, contours, hierarchy = cv2.findContours(bin_frame, cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE)

    hierarchy = hierarchy[0] # Why does this list have a 3rd dimension? Let's remove it
    # for i, val in enumerate(hierarchy):
    #     print(i, val)

    # Because the area surrounding the percentage is white, the bounding box of the entire frame counts as a contour
    # print(list(map(cv2.contourArea, contours)))
    bounding_box_idx = np.argsort(list(map(cv2.contourArea, contours)))[-1]

    # Loop over contours, saving only those that belong to the percentage (i.e. not the bounding box, a child of the
    # bounding box, or a hole in one of the numerals)
    percentage_contours = []
    for i, contour in enumerate(contours):
        if i != bounding_box_idx and hierarchy[i][3] != bounding_box_idx:
            if not (hierarchy[i][3] == -1 and cv2.contourArea(contour) < 0.01 * frame.size):
                percentage_contours.append(contour)
    cv2.drawContours(frame, percentage_contours, -1, 255, 1)

    bin_alt = np.zeros(bin_frame.shape, np.uint8)
    cv2.drawContours(bin_alt, percentage_contours, -1, 255, cv2.FILLED)

    if True:
        cv2.imshow('Binarized and denoised', bin_alt)
        cv2.waitKey(0)

    # bin_frame = cv2.imread('bintest.jpg', flags=0)
    # for i, contour in enumerate(contours):
    #     bin_contour = np.array(frame, np.uint8)
    #     cv2.drawContours(bin_contour, [contour], 0, 255, 1)
        # cv2.imshow('contour', bin_contour)
        # cv2.waitKey(0)
    cv2.imwrite('bin.jpg', bin_alt)
    show_ocr_result(bin_alt, '7')

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
