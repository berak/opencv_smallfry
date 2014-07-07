/*
    Copyright 2012 Computer Vision Lab,
    Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
    All rights reserved.

    Authors: Tomasz Trzcinski, Eray Molla, and Vincent Lepetit

    This file is part of the DBRIEF_demo software.

    DBRIEF_demo is  free software; you can redistribute  it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free  Software Foundation; either  version 2 of the  License, or
    (at your option) any later version.

    DBRIEF_demo is  distributed in the hope  that it will  be useful, but
    WITHOUT  ANY   WARRANTY;  without  even  the   implied  warranty  of
    MERCHANTABILITY  or FITNESS FOR  A PARTICULAR  PURPOSE. See  the GNU
    General Public License for more details.

    You should  have received a copy  of the GNU  General Public License
    along  with DBRIEF_demo;  if  not,  write  to   the  Free  Software
    Foundation,  Inc.,  51  Franklin  Street, Fifth  Floor,  Boston,  MA
    02110-1301, USA
*/

#pragma once

#include <vector>
#include <bitset>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;

namespace CVLAB 
{
    // Length of the dbrief descriptor:
    static const int DESC_LEN = 32;

    // Size of the area in which the tests are computed:
    static const int PATCH_SIZE = 32;

    // Box kernel size used for smoothing. Must be an odd positive integer:
    static const int KERNEL_SIZE = 5;

    // Size of the area surrounding feature point
    static const int PATCH_SIZE_2 = PATCH_SIZE * PATCH_SIZE;
    // Half of the patch size
    static const int HALF_PATCH_SIZE = PATCH_SIZE >> 1;
    // Area of the smoothing kernel
    static const int KERNEL_AREA = KERNEL_SIZE * KERNEL_SIZE;
    // Half of the kernel size
    static const int HALF_KERNEL_SIZE = KERNEL_SIZE >> 1;

    /***************************************************************************************************
    *
    *  +++++++++++++++++++
    *  +-a-            | +
    *  +               b +       a = IMAGE_PADDING_LEFT
    *  +               | +       b = IMAGE_PADDING_TOP
    *  +   +++++++++++   +       The area inside is the subimage we look for the keypoints such that
    *  +   +         +   +       we avoid the border effect of the smoothing.
    *  +   +         +   +       Note that in our implementation a = b
    *  +   +         +   +
    *  +   +         +   +
    *  +   +++++++++++   +
    *  + |            -a-+
    *  + b               +
    *  + |               +
    *  +++++++++++++++++++
    *  Figure 1
    *
    **************************************************************************************************/

    // See figure above:
    static const int IMAGE_PADDING_TOP = HALF_KERNEL_SIZE + HALF_PATCH_SIZE;
    static const int IMAGE_PADDING_LEFT = IMAGE_PADDING_TOP;
    static const int IMAGE_PADDING_TOTAL = IMAGE_PADDING_TOP << 1;
    static const int IMAGE_PADDING_RIGHT = IMAGE_PADDING_LEFT;
    static const int IMAGE_PADDING_BOTTOM = IMAGE_PADDING_TOP;
    static const int SUBIMAGE_LEFT = IMAGE_PADDING_LEFT;
    static const int SUBIMAGE_TOP = IMAGE_PADDING_TOP;

    // Returns the Hamming Distance between two dbrief descriptors
    inline int HAMMING_DISTANCE(const bitset<DESC_LEN>& d1, const bitset<DESC_LEN>& d2)
    {
    return (d1 ^ d2).count();
    }

    // Returns the width of the subimage shown in the figure above given the original image width:
    inline int SUBIMAGE_WIDTH(const int width)
    {
        return width - IMAGE_PADDING_TOTAL;
    }

    // Returns the width of the subimage shown in the figure above given the original image width:
    inline int SUBIMAGE_HEIGHT(const int height)
    {
        return height - IMAGE_PADDING_TOTAL;
    }

    // Returns the x-coordinate of the right edge of the subimage
    inline int SUBIMAGE_RIGHT(const int width) 
    {
        return width - IMAGE_PADDING_RIGHT;
    }

    // Returns the y-coordinate of the bottom edge of the subimage
    inline int SUBIMAGE_BOTTOM(const int height) 
    {
        return height - IMAGE_PADDING_BOTTOM;
    }

    // A class which represents the operations of Dbrief keypoint descriptor
    class Dbrief {
    public:

        // Given keypoint kpt and image img, returns the dbrief descriptor desc
        void getDbriefDescriptor(bitset<DESC_LEN>& desc, cv::KeyPoint kpt, const cv::Mat & img);

        // Given keypoints kpts and image img, returns dbrief descriptors descs
        void getDbriefDescriptors(vector< bitset<DESC_LEN> >& descriptors, const vector<cv::KeyPoint>& kpts, const cv::Mat & img);

    private:

        // Allocate space for storing box smoothed image
        void allocateBoxSmoothedImage(const cv::Mat & img);

        // Checks if the tests locations for the keypoints in kpts lie inside an im_w x im_h image:
        bool validateKeypoints(const vector< cv::KeyPoint >& kpts, int im_w, int im_h);

        // Returns true if kpt is inside the subimage
        bool isKeypointInsideSubImage(const cv::KeyPoint& kpt, const int width, const int height);

        // the image smoothed with a box filter
        cv::Mat boxSmoothedImage;

    };

};
