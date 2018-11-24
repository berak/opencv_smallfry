import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.*;

public class FaceDetect {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat img = Imgcodecs.imread("../img/face.jpg"); // your data here !
        Net net = Dnn.readNetFromTensorflow("c:/data/mdl/opencv_face_detector_uint8.pb","c:/data/mdl/opencv_face_detector.pbtxt");
        // you want a downscaled Size of your input img, but keep the aspect ratio. 128x96 is 1/5 of a vga img
        Mat inputBlob = Dnn.blobFromImage(img, 1.0f, new Size(128,96), new Scalar(104, 177, 123, 0), false, false);
        net.setInput(inputBlob);
        Mat res = net.forward("detection_out");
        // the net's output is a list of [n#,id,conf,t,l,b,r], one row per face found
        Mat faces = res.reshape(1, res.size(2));
		System.out.println("faces" + faces);
        float [] data = new float[7];
        for (int i=0; i<faces.rows(); i++)
        {
            faces.get(i, 0, data);
            float confidence = data[2];
            if (confidence > 0.4f)
            {
                int left   = (int)(data[3] * img.cols());
                int top    = (int)(data[4] * img.rows());
                int right  = (int)(data[5] * img.cols());
                int bottom = (int)(data[6] * img.rows());
                System.out.println("("+left + "," + top + ")("+right+","+bottom+") " + confidence);
        		Imgproc.rectangle(img, new Point(left,top), new Point(right,bottom), new Scalar(0,200,0), 3);
            }
        }
        Imgcodecs.imwrite("facedet.png", img);
    }
}
