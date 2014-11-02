import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.*;
import java.util.*;

import lowgui.*;
import texture.*;

class SimpleSample {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static Size cropSize = new Size(90,90);
    public static Mat loadImg(String name) {
        Mat m = Imgcodecs.imread(name,0);
        Imgproc.resize(m,m,cropSize);
        //System.err.println(name + " " + m);
        return m;
    }

    public static void main(String[] args) {
        /*
        lowgui.NamedWindow frame = new lowgui.NamedWindow("Jolly");
        Mat im = Imgcodecs.imread("dogwalker.jpg");
        frame.setSize(im.cols(),im.rows());
        frame.imshow(im);
        frame.waitKey(-1);
        */
        String [] names = {"aRUtJ.png","img.jpg","right01.jpg","phone.jpg","right01.jpg"};
        List<Mat> data = new ArrayList<Mat>();
        for (int i=0; i<5; i++) {
            data.add(loadImg("e:/code/opencv_p/demo/" + names[i]));
        }
        data.add(loadImg("dogwalker.jpg"));
        Mat labels = new MatOfInt(1,2,3,4,5,17);

        texture.FaceRec face = new texture.FaceRec("Lbph","Cos");
        face.train(data,labels);

        Mat img = loadImg("dogwalker.jpg");
        Mat res = face.predict(img);
        System.err.println(res.dump());
    }
}
