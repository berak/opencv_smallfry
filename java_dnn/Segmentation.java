import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.*;

public class Segmentation {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	    Scalar[] colors = new Scalar[]{ // obvious TODO: find better colours ;)
	    	new Scalar(1,22,123), new Scalar(28,53,4), new Scalar(81,2,243), new Scalar(2,223,244),
	    	new Scalar(12,2,3), new Scalar(2,3,4), new Scalar(161,162,3), new Scalar(124,13,144),
	    	new Scalar(112,2,223), new Scalar(2,3,4), new Scalar(81,82,3), new Scalar(222,223,24),
	    	new Scalar(12,222,32), new Scalar(82,3,4), new Scalar(221,12,3), new Scalar(182,83,84),
	    	new Scalar(221,12,3), new Scalar(52,3,4), new Scalar(91,162,23), new Scalar(211,3,49)
	    };
        Mat img = Imgcodecs.imread("c:/data/img/warp.png"); // your data here !
        Net net = Dnn.readNet("c:/data/mdl/enet-model-best.net");
        Mat inputBlob = Dnn.blobFromImage(img, 1.0/255, new Size(512,256), new Scalar(0, 0, 0), true, false);
        net.setInput(inputBlob);
        Mat res = net.forward();
        // the net's output is a list of "probability maps", one per class (size(1) is, how many classes there are, 2 and 3 are H,W)
        System.out.println(res.size(0) + " " + res.size(1) + " " + res.size(2) + " " + res.size(3));
        Mat strip = res.reshape(1, res.size(1) * res.size(2)); // make a long, vertical "image strip" of it (since we can't access 4d tensors from java)
        Mat probs = new Mat(res.size(2), res.size(3), res.type(), Scalar.all(0)); // will keep "highest probability" per pixel
        Mat segm  = new Mat(probs.size(), CvType.CV_8UC3); // color overlay
        // check each map, keep the pixel with highest probability and (re)assign a color to it
        for (int i=0; i<res.size(1); i++) {
        	Mat sub = strip.submat(i*res.size(2), (i+1)*res.size(2), 0, res.size(3)); // probs for class i
        	// find out, which pixels int the current map had a higher probability, than all of the prev. ones
        	Mat gt = new Mat(); // 'greater than all' mask
        	Core.compare(sub, probs, gt, Core.CMP_GT);
        	// update probs and segmentation
        	segm.setTo(colors[i], gt);
        	sub.copyTo(probs, gt);
        }
        Mat result = new Mat();
        Imgproc.resize(segm, segm, img.size());
        Core.addWeighted(img, 0.8, segm, 0.2, 0, result); // alpa + beta == 1
        Imgcodecs.imwrite("segm.png", result);
    }
}
