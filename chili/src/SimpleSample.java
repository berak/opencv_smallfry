
import java.util.*;
import org.opencv.calib3d.*;
import org.opencv.core.*;
import org.opencv.face.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.*;
import org.opencv.objdetect.*;
import org.opencv.photo.*;
import org.opencv.utils.*;
import org.opencv.video.*;
import org.opencv.videoio.*;


class SimpleSample {
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    public static void cout(Object ... s){ for(Object z:s)System.out.println(z); }
    public static void cerr(Object ... s){ for(Object z:s)System.err.println(z); }
    public static void help(){ cerr("help(classname,item);\n  'classname' should be canonical, like org.opencv.core.Mat\n  'item' can be: CONSTRUCTOR, FIELD, METHOD, CLASS, ALL"); }
    public static void help(String cls){ ClassSpy.reflect(cls,"CLASS"); }
    public static void help(String cls,String item){ ClassSpy.reflect(cls,item); }
    public static void main(String[] args) {

        Mat ocv = Imgcodecs.imread("input.img",-1);
        if ( ocv.empty() )
            ocv = new Mat(8,8,CvType.CV_8UC3,new Scalar(40,40,40));
help("org.opencv.face.Face","ALL");
        ;;
        Imgcodecs.imwrite("output.png", ocv);
        System.exit(0); // to break out of the ant shell.
    }
}
