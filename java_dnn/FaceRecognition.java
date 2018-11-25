import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.*;

public class FaceRecognition {
	public static Mat process(Net net, Mat img) {
        Mat inputBlob = Dnn.blobFromImage(img, 1./255, new Size(96,96), new Scalar(0,0,0), true, false);
        net.setInput(inputBlob);
        return net.forward().clone();
	}
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Net net = Dnn.readNetFromTorch("c:/data/mdl/openface.nn4.small2.v1.t7");
        Mat feature1 = process(net, Imgcodecs.imread("C:/data/faces/lfw40_crop/Abdullah_Gul_0004.jpg")); // your data here !
        Mat feature2 = process(net, Imgcodecs.imread("C:/data/faces/lfw40_crop/Abdullah_Gul_0007.jpg")); // your data here !
        double dist  = Core.norm(feature1,  feature2);
        if (dist < 0.6)
        	System.out.println("SAME !");
    }
}
