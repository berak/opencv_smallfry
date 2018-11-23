import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.*;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.*;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.*;

public class SimpleSample {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        float scoreThresh = 0.5f;
        float nmsThresh = 0.4f;
        // Model from https://github.com/argman/EAST
        // You can find it here : https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py#L309
        Net net = Dnn.readNetFromTensorflow("c:/data/mdl/frozen_east_text_detection.pb");
        // input image
        Mat frame = Imgcodecs.imread("text2.png");
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        Size siz = new Size(320, 320);
        int W = (int)(siz.width / 4);
        int H = (int)(siz.height / 4);
        Mat blob = Dnn.blobFromImage(frame, 1.0,siz, new Scalar(123.68, 116.78, 103.94), true, false);
        net.setInput(blob);
        List<Mat> outs = new ArrayList<>(2);
        List<String> outNames = new ArrayList<String>();
        outNames.add("feature_fusion/Conv_7/Sigmoid");
        outNames.add("feature_fusion/concat_3");
        net.forward(outs, outNames);
        // scores.shape : (1,1,80,80)
        Mat scores = outs.get(0).reshape(1, H);
        // geometry.shape : (1,5,80,80)
        // My lord and savior : http://answers.opencv.org/question/175676/javaandroid-access-4-dim-mat-planes/
        Mat geometry = outs.get(1).reshape(1, 5 * H);
        List<Float> confidencesList = new ArrayList<>();
        // Decode predicted bounding boxes.
        List<RotatedRect> boxesList = decode(scores, geometry, confidencesList, scoreThresh);

        MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confidencesList));
        RotatedRect[] boxesArray = boxesList.toArray(new RotatedRect[0]);
        MatOfRotatedRect boxes = new MatOfRotatedRect(boxesArray);
        MatOfInt indices = new MatOfInt();
        // Apply non-maximum suppression procedure.
        Dnn.NMSBoxesRotated(boxes, confidences, scoreThresh, nmsThresh, indices);

        // Render detections
        Point ratio = new Point((float)frame.cols()/siz.width, (float)frame.rows()/siz.height);
        int[] indexes = indices.toArray();
        for(int i = 0; i<indexes.length;++i) {
            RotatedRect rot = boxesArray[indexes[i]];
            Point[] vertices = new Point[4];
            rot.points(vertices);
            for (int j = 0; j < 4; ++j) {
                vertices[j].x *= ratio.x;
                vertices[j].y *= ratio.y;
            }
            for (int j = 0; j < 4; ++j) {
                Imgproc.line(frame, vertices[j], vertices[(j + 1) % 4], new Scalar(0, 0,255), 2);
            }
        }
        Imgcodecs.imwrite("out.png", frame);
    }

    private static List<RotatedRect> decode(Mat scores, Mat geometry, List<Float> confidences, float scoreThresh) {
        // size of 1 geometry plane
        int W = geometry.cols();
        int H = geometry.rows() / 5;
        System.out.println(geometry);
        System.out.println(scores);

        List<RotatedRect> detections = new ArrayList<>();
        for (int y = 0; y < H; ++y) {
            Mat scoresData = scores.row(y);
            //1st plane
            Mat x0Data = geometry.submat(0, H, 0, W).row(y);
            //2nd plane
            Mat x1Data = geometry.submat(H, 2 * H, 0, W).row(y);
            Mat x2Data = geometry.submat(2 * H, 3 * H, 0, W).row(y);
            Mat x3Data = geometry.submat(3 * H, 4 * H, 0, W).row(y);
            Mat anglesData = geometry.submat(4 * H, 5 * H, 0, W).row(y);

            for (int x = 0; x < W; ++x) {
                double score = scoresData.get(0, x)[0];
                if (score >= scoreThresh) {
                    double offsetX = x * 4.0;
                    double offsetY = y * 4.0;
                    double angle = anglesData.get(0, x)[0];
                    double cosA = Math.cos(angle);
                    double sinA = Math.sin(angle);
                    double x0 = x0Data.get(0, x)[0];
                    double x1 = x1Data.get(0, x)[0];
                    double x2 = x2Data.get(0, x)[0];
                    double x3 = x3Data.get(0, x)[0];
                    double h = x0 + x2;
                    double w = x1 + x3;
                    Point offset = new Point(offsetX + cosA * x1 + sinA * x2, offsetY - sinA * x1 + cosA * x2);
                    Point p1 = new Point(-1 * sinA * h + offset.x, -1 * cosA * h + offset.y);
                    Point p3 = new Point(-1 * cosA * w + offset.x,      sinA * w + offset.y);
                    RotatedRect r = new RotatedRect(new Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y)), new Size(w, h), -1 * angle * 180 / Math.PI);
                    detections.add(r);
                    confidences.add((float) score);
                }
            }
        }
        return detections;
    }
}
