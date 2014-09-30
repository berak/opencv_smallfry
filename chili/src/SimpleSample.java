
import org.opencv.core.*;
import org.opencv.highgui.*;
import org.opencv.imgproc.*;
import org.opencv.video.*;
import org.opencv.objdetect.*;

class SimpleSample {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        Mat ocv = Highgui.imread("input.img",-1);
        if ( ocv.empty() )
            ocv = new Mat(8,8,CvType.CV_8UC3,new Scalar(40,40,40));

        ;;
        Highgui.imwrite("output.png", ocv);
        System.exit(0); // to break out of the ant shell.
    }
}
