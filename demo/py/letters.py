import cv2
import numpy as np

def get_cnt_area(cnt):
    rect = cv2.minAreaRect(cnt)
    center, width_height, angle = rect
    area = width_height[0] * width_height[1]

    return area


def get_correct_cnt(cnts, img):
    height, width = img.shape
    area = height * width

    for cnt in cnts:
        letter_area = get_cnt_area(cnt)
        if letter_area > area / 10 and letter_area < area * 9 / 10:
            pdb.set_trace()
            return cnt

    return None


def shape_match():

    img1 = cv2.imread('l1.png', 0)
    img2 = cv2.imread('l2.png', 0)

    ret, thresh1 = cv2.threshold(img1, 13, 255, 1)
    ret, thresh2 = cv2.threshold(img2, 13, 255, 1)

    im1,contours1,hierarchy = cv2.findContours(thresh1, 2, 1)
    im2,contours2,hierarchy = cv2.findContours(thresh2, 2, 1)
    print len(contours1), len(contours2)
    cv2.drawContours(img1,contours1, -1, (200,0,0))
    cv2.drawContours(img2,contours2, -1, (200,0,0))
    cv2.imshow("1",img1)
    cv2.imshow("2",img2)
    cv2.waitKey()
    #query_cnt = get_correct_cnt(contours2, img2)
    #if query_cnt is None:
    #    print 'parse img failed'
    #    return

    height, width  = img1.shape
    area = height * width
    min_area = area / 25
    max_area = area / 5

    for cnt in contours1:
        print cv2.boundingRect(cnt)
        letter_area = get_cnt_area(cnt)
        if not (min_area < letter_area and letter_area < max_area):
            continue

        print cv2.matchShapes(cnt, query_cnt, 1, 0.0)

if __name__ =='__main__':
    shape_match()
