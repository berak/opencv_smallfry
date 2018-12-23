import org.opencv.core.Core;
import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.utils.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class StyleTransfer  {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // model taken from:
        // https://github.com/jcjohnson/fast-neural-style/blob/master/models/download_style_transfer_models.sh
        String modelWeights = "c:/data/mdl/udnie.t7";
        Net net = Dnn.readNet(modelWeights);

        Mat image = Imgcodecs.imread("dog.jpg");
        Imgproc.resize(image, image, new Size(700,480));

        Scalar mean = new Scalar(103.939, 116.779, 123.680);
        Mat blob = Dnn.blobFromImage(image, 1.0, new Size(), mean, false, false);
        net.setInput(blob);
        Mat result = net.forward();

        int H = result.size(2);
        int W = result.size(3);

        // step 1: reshape it to a long vertical strip:
        Mat strip = result.reshape(1, H * 3);

        // step 2: collect the color planes into a list:
        List<Mat> lis = new ArrayList<>();
        lis.add(strip.submat(0,H, 0,W));
        lis.add(strip.submat(H,2*H, 0,W));
        lis.add(strip.submat(2*H,3*H, 0,W));

        // step 3: merge planes into final bgr image
        Mat bgr = new Mat();
        Core.merge(lis, bgr);

        // last: add the mean value
        Core.add(bgr, mean, bgr);

        Imgcodecs.imwrite("out.png", bgr);
    }
}
