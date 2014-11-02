package texture;

import org.opencv.core.*;
import org.opencv.imgproc.*;

import java.util.*;

public abstract class Extractor {
	public abstract Mat extract(Mat in);
	public abstract Mat extract(List<Mat> imglist);

	static Extractor create(String s) {
		if (s.startsWith("Lbp"))
			return new Lbph();
		return new Pixels();
	}
}


class Pixels extends Extractor {
	public Mat extract(Mat img) {
		if (img.type() != CvType.CV_8U) {
			Imgproc.cvtColor(img,img,Imgproc.COLOR_BGR2GRAY);
		}
		return img;
	}
	public Mat extract(List<Mat> imglist) {
		Mat feat = new Mat();
		for ( Mat m :imglist ) {
			feat.push_back(extract(m));
		}
		return feat;
	}
}

class Lbph extends Pixels {
	public Mat extract(Mat img) {
		Mat I = super.extract(img);
        int M = I.rows(); 
        int N = I.cols(); 
        int h = M/8;
        int w = N/8;
        short [] his = new short[256*8*8];
        byte  [] px  = new byte[M*N];
        I.get(0,0,px);

        for (int i=1; i<M-h; i++) {
	       	int oi = i/h;
	        for (int j=1; j<N-w; j++) {
	        	byte i7 = px[(j-1)*N + (i-1)];
	        	byte i6 = px[(j-1)*N + (i  )];
	        	byte i5 = px[(j-1)*N + (i+1)];
	        	byte i4 = px[(j  )*N + (i+1)];
	        	byte i3 = px[(j+1)*N + (i+1)];
	        	byte i2 = px[(j+1)*N + (i  )];
	        	byte i1 = px[(j+1)*N + (i-1)];
	        	byte i0 = px[(j  )*N + (i-1)];
	        	byte ic = px[(j  )*N + (i  )];
	        	int  code = (i7>ic ? 1<<7 : 0) 
	        	          + (i6>ic ? 1<<6 : 0)
	        	          + (i5>ic ? 1<<5 : 0)
	        	          + (i4>ic ? 1<<4 : 0)
           	        	  + (i3>ic ? 1<<3 : 0)
	                      + (i2>ic ? 1<<2 : 0)
	        	          + (i1>ic ? 1<<1 : 0)
	        	          + (i0>ic ? 1<<0 : 0) ;
		       	int oj  = j/w;
		       	int off = 256*(oi*8+oj);
	        	his[off+code] ++;
	        }
        }
        Mat hist = new Mat(1,256*8*8,CvType.CV_16S);
        hist.put(0,0,his);
		return hist;
	}
}

/*
class Mts extends Pixels {
	public Mat extract(Mat img) {
		Mat I = super.extract(img);
        int M = I.rows(); 
        int N = I.cols(); 
        Mat IC = new Mat(I,new Range(2,M-1), new Range(2,N-1));
		Mat I7 = new Mat(I,new Range(1,M-2), new Range(1,N-2));
		Mat I6 = new Mat(I,new Range(1,M-2), new Range(2,N-1));
		Mat I5 = new Mat(I,new Range(1,M-2), new Range(3,N  ));
		Mat I4 = new Mat(I,new Range(2,M-1), new Range(3,N  ));
		Mat ret = new Mat();
// yea, forget that..
//        fI = ((IC>=I7)&8) | ((IC>=I6)&4) | ((IC>=I5)&2) | ((IC>=I4)&1);
		return ret;
	}
}
*/
