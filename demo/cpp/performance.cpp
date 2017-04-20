/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
 * performance.cpp
 *
 * Measure performance of classifier
 *
 * !! this code assumes, you have absolute path to your images in the info file.
 */

#include "opencv2/opencv.hpp"

#include <cstdio>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;

struct ObjectPos
{
    float x;
    float y;
    float width;
    int found;    // for reference
    int neighbors;
};

int main( int argc, char* argv[] )
{
    CommandLineParser parser(argc, argv,
        "{ help about h ? |   | print this help page }"
        "{ data d         |   | (required) trained classifier xml file }"
        "{ info i         |   | (required) info (collection) file name }"
        "{ saveImage S    |   | save image with found rects }"
        "{ maxSizeDiff  m |1.5| maximum size difference allowed }"
        "{ maxPosDiff p   |0.3| maximum position difference allowed }"
        "{ scale s        |1.2| scale factor for cascade detection }"
        "{ rocSize r      |100| number of points used for ROC }" );

    String cascadeXML(parser.get<String>("data"));
    String infoname(parser.get<String>("info"));
    double scale_factor(parser.get<double>("scale"));
    float maxSizeDiff(parser.get<double>("maxSizeDiff"));
    float maxPosDiff(parser.get<double>("maxPosDiff"));
    int rocsize(parser.get<int>("rocSize"));
    bool saveDetected = parser.has("i");
    if (parser.has("help") || infoname.empty() || cascadeXML.empty())
    {
        parser.printMessage();
        return 0;
    }

    CascadeClassifier cascade(cascadeXML);
    if (cascade.empty())
    {
        cout << "Unable to load classifier from " << cascadeXML << endl;
        return 1;
    }

    ifstream info( infoname );
    if ( ! info.good() )
    {
        cout << "Unable to load info file from " << infoname << endl;
        return 1;
    }
    printf( "+================================+======+======+======+\n" );
    printf( "|            File Name           | Hits |Missed| False|\n" );
    printf( "+================================+======+======+======+\n" );

    double totaltime = 0;
    vector<int> pos(rocsize, 0), neg(rocsize, 0);
    int hits = 0, totalHits = 0, missed = 0, totalMissed = 0, falseAlarms = 0, totalFalseAlarms = 0;
    while ( info.good() )
    {
        string filename;
        int refcount;
        info >> filename >> refcount;
        if ( ! info.good() ) break;
        if ( refcount <= 0 ) continue;

        Mat img = imread( filename );
        if ( img.empty() )
        {
            printf("image %s not found !\n", filename.c_str());
            continue;
        }

        int error = 0;
        vector<ObjectPos> ref(refcount);
        for ( int i = 0; i < refcount; i++ )
        {
            int x, y, w, h;
            info >> x >> y >> w >> h;
            if ( ! info.good() ) return -1;
            ref[i].x = 0.5F * w + x;
            ref[i].y = 0.5F * h + y;
            ref[i].width = sqrtf( 0.5F * (w * w + h * h) );
            ref[i].found = 0;
            ref[i].neighbors = 0;
        }
        if ( error )
        {
            printf("error parsing info file !\n");
            break;
        }

        totaltime -= time( 0 );
        vector<Rect> objects;
        vector<int> counts;
        cascade.detectMultiScale( img, objects, counts, scale_factor, 1 );
        totaltime += time( 0 );
        hits = missed = falseAlarms = 0;
        vector<ObjectPos> det(objects.size());
        for ( size_t i = 0; i < objects.size(); i++ )
        {
            Rect r = objects[i];
            det[i].x = 0.5F * r.width  + r.x;
            det[i].y = 0.5F * r.height + r.y;
            det[i].width = sqrtf( 0.5F * (r.width * r.width +
                                          r.height * r.height) );
            det[i].neighbors = counts[i];

            if ( saveDetected )
            {
                rectangle( img, Point( r.x, r.y ),
                    Point( r.x + r.width, r.y + r.height ),
                    Scalar( 0, 0, 255 ), 3 );
            }

            int found = 0;
            for ( int j = 0; j < refcount; j++ )
            {
                double distance = sqrtf( (det[i].x - ref[j].x) * (det[i].x - ref[j].x) +
                                         (det[i].y - ref[j].y) * (det[i].y - ref[j].y) );
                if ( (distance < ref[j].width * maxPosDiff) &&
                     (det[i].width > ref[j].width / maxSizeDiff) &&
                     (det[i].width < ref[j].width * maxSizeDiff) )
                {
                    ref[j].found = 1;
                    ref[j].neighbors = MAX( ref[j].neighbors, det[i].neighbors );
                    found = 1;
                }
            }
            if ( !found )
            {
                falseAlarms++;
                neg[MIN(det[i].neighbors, rocsize - 1)]++;
            }
        }
        for ( int j = 0; j < refcount; j++ )
        {
            if ( ref[j].found )
            {
                hits++;
                pos[MIN(ref[j].neighbors, rocsize - 1)]++;
            }
            else
            {
                missed++;
            }
        }

        totalHits += hits;
        totalMissed += missed;
        totalFalseAlarms += falseAlarms;
        printf( "|%32.32s|%6d|%6d|%6d|\n", filename.c_str(), hits, missed, falseAlarms );
        printf( "+--------------------------------+------+------+------+\n" );
        fflush( stdout );

        if ( saveDetected )
        {
            imwrite( format("%s.det.png", filename.c_str()), img );
        }
        imshow("perf", img);
        waitKey();
    }

    printf( "|%32.32s|%6d|%6d|%6d|\n", "Total",
            totalHits, totalMissed, totalFalseAlarms );
    printf( "+================================+======+======+======+\n" );
    // printf( "Number of stages: %d\n", nos );
    // printf( "Number of weak classifiers: %d\n", numclassifiers[nos - 1] );
    printf( "Total time: %f\n", totaltime );

    // print ROC to stdout
    for ( int i = rocsize - 1; i > 0; i-- )
    {
        pos[i-1] += pos[i];
        neg[i-1] += neg[i];
    }
    for ( int i = 0; i < rocsize; i++ )
    {
        fprintf( stderr, "\t%d\t%d\t%f\t%f\n", pos[i], neg[i],
            ((float)pos[i]) / (totalHits + totalMissed),
            ((float)neg[i]) / (totalHits + totalMissed) );
    }

    return 0;
}
