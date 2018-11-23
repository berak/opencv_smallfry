import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.*;

public class Pose {

    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat img = Imgcodecs.imread("c:/data/img/persons/single2.png");
        // read the network model
        Net net = Dnn.readNetFromTensorflow("c:/data/mdl/body/tf_small.pb");
        //Net net = Dnn.readNetFromCaffe("c:/data/mdl/body/openpose_pose_coco.prototxt", "c:/data/mdl/body/pose_iter_440000.caffemodel");

        // send it through the network
        Mat inputBlob = Dnn.blobFromImage(img, 1.0, new Size(368,368), new Scalar(0, 0, 0), false, false);
        // Mat inputBlob = Dnn.blobFromImage(img, 1.0 / 255, new Size(368,368), new Scalar(0, 0, 0), false, false);
        net.setInput(inputBlob);
        //Mat result = net.forward().reshape(1,19); // 19 body parts
        Mat result = net.forward().reshape(1,57); // 19 body parts + 2 * 19 PAF maps

        System.out.println(result);

        // get the heatmap locations
        ArrayList<Point> points = new ArrayList();
        for (int i=0; i<18; i++) { // skip background
            Mat heatmap = result.row(i).reshape(1,46); // 46x46
            Core.MinMaxLocResult mm = Core.minMaxLoc(heatmap);
            Point p = new Point();
            if (mm.maxVal>0.1f) {
                p = mm.maxLoc;
            }
            points.add(p);
            System.out.println(i + " " + p + " " + heatmap);
        }

        // 17 possible limb connections
        int pairs[][] = {
            {1,2}, {1,5}, {2,3},
            {3,4}, {5,6}, {6,7},
            {1,8}, {8,9}, {9,10},
            {1,11}, {11,12}, {12,13},
            {1,0}, {0,14},
            {14,16}, {0,15}, {15,17}
        };

        // connect body parts and draw it !
        float SX = (float)(img.cols()) / 46;
        float SY = (float)(img.rows()) / 46;
        for (int n=0; n<17; n++)
        {
            // lookup 2 connected body/hand parts
            Point a = points.get(pairs[n][0]).clone();
            Point b = points.get(pairs[n][1]).clone();

            // we did not find enough confidence before
            if (a.x<=0 || a.y<=0 || b.x<=0 || b.y<=0)
                continue;

            // scale to image size
            a.x*=SX; a.y*=SY;
            b.x*=SX; b.y*=SY;

            Imgproc.line(img, a, b, new Scalar(0,200,0), 2);
            Imgproc.circle(img, a, 3, new Scalar(0,0,200), -1);
            Imgproc.circle(img, b, 3, new Scalar(0,0,200), -1);
        }
        Imgcodecs.imwrite("pose.png", img);
    }
}
