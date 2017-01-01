
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int err(String message) {
    cout << message << endl;
    return (1);
}

int main()
{
    VideoCapture playback("A.avi"); // playback video
    if (!playback.isOpened()) return err("playback did not open");
    VideoCapture cam(0); // camera
    if (!cam.isOpened()) return err("camera did not open");

    // create GUI windows
    namedWindow("Frame");
    namedWindow("Video Frame");

    // please *avoid* Global variables
    Mat cam_frame, pb_frame; // current frames
    int keyboard = 0; // input from keyboard (raii)
    bool condition = false; // ???
    while ((char)keyboard != 'q' && (char)keyboard != 27) {
        cam >> cam_frame;
        if (cam_frame.empty())
            break;

        flip(cam_frame, cam_frame, 1);

        // no idea, what you have in mind, i'll use the space bar:
        if (keyboard==' ') condition = ! condition;
        if (condition) {
            playback >> pb_frame;
            if (pb_frame.empty()) { // end of movie, what now ? just rewind ?
                playback.set(CAP_PROP_POS_FRAMES, 0);
            } else { // only show, if valid
                imshow("Video Frame", pb_frame);
            }
        }

        imshow("Frame", cam_frame);
        keyboard = waitKey(30);
    }

    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}
