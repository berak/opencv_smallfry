package texture;

import org.opencv.core.*;
import org.opencv.imgproc.*;


public abstract class Classifier {
    // dataset should have one row per feature, 
    // labels should have n rows, 1 classid per features
    public abstract boolean train(Mat dataset, Mat labels);

    // return MatOfFloat( classid, distance, imgid ) ;
    public abstract Mat predict(Mat query); 

    static Classifier create(String s) {
        if ( s.startsWith("Hist") )
            return new CompareHist();
        if ( s.startsWith("Cos") )
            return new CompareCosine();
        if ( s.startsWith("Eigen") )
            return new CompareEigen();
        return new CompareNorm();
    }

    // ubitiquous helper:
    public Mat tofloat(Mat in) {
        if (in.type() != CvType.CV_32F) {
            in.convertTo(in, CvType.CV_32F);            
        }
        return in;
    }    
}



class CompareNorm extends Classifier {
    Mat features;
    Mat labels;
    int flag=Core.NORM_L2;

    public CompareNorm() {}
    public CompareNorm(int flag) {
        this.flag=flag;
    }
    public boolean train(Mat in, Mat labels) {
        this.features = in;
        this.labels = labels;
        return true;
    }
    public double distance(Mat a, Mat b) {
        return Core.norm(a,b,flag);
    }
    public Mat predict(Mat in) {
        double dm = 99999999.0;
        double best = -1; 
        int di = -1;
        for ( int i=0; i<features.rows(); i++) {
            double d = distance(in,features.row(i));
            if ( d<dm ) {
                di = i;
                dm = d;
                best = labels.get(di,0)[0];
            }
        }
        return new MatOfFloat((float)best,(float)dm,(float)di);
    }
}


class CompareFloat extends CompareNorm {
    public boolean train(Mat in, Mat labels) {
        return super.train(tofloat(in), labels);
    }
    public Mat predict(Mat in) {
        return super.predict(tofloat(in));
    }
}

class CompareHist extends CompareFloat {
    public CompareHist() {
        this.flag=Imgproc.CV_COMP_HELLINGER;
    }
    public CompareHist(int flag) {
        this.flag=flag;
    }
    public double distance(Mat a, Mat b) {
        return Imgproc.compareHist(a,b,flag);
    }
}


class CompareCosine extends CompareFloat {
    public double distance(Mat trainFeature, Mat testFeature) {
        double a = trainFeature.dot(testFeature);
        double b = trainFeature.dot(trainFeature);
        double c = testFeature.dot(testFeature);
        return -a / Math.sqrt(b*c);
    }
};



class CompareEigen extends Classifier
{
    Mat _projections = new Mat();;
    Mat _labels = new Mat();;
    Mat _eigenvectors = new Mat();;
    Mat _mean = new Mat();;
    int _num_components = 10;

    public CompareEigen() {}
    public CompareEigen(int num_components) {
        _num_components = (num_components); // we don't need a threshold yet
    }

    public void save_projections(Mat data) {
        _projections = new Mat();
        for(int i=0; i<data.rows(); i++) {
            
            Mat p = new Mat();
            Core.PCAProject(_eigenvectors, _mean, data.row(i), p);
            _projections.push_back(p);
        }
    }

    public boolean train(Mat data, Mat labels) {
        Mat fd = tofloat(data);
        if((_num_components <= 0) || (_num_components > fd.rows()))
            _num_components = fd.rows();
        
        Mat eigenvectors = new Mat();
        Mat mean = new Mat();
        Core.PCACompute(fd, mean, eigenvectors, _num_components);

        _labels = labels;
        _mean = mean.reshape(1, fd.cols());
        Core.transpose(eigenvectors, _eigenvectors);
        save_projections(fd);
        return true;
    }

    public Mat predict(Mat testFeature) {
        double minDist = 9999999.0;
        double minClass = -1;
        int minId=-1;
        Mat query = new Mat();
        Core.PCAProject(_eigenvectors, _mean, tofloat(testFeature), query);
        for (int idx=0; idx<_projections.rows(); idx++) {
            double dist = Core.norm(_projections.row(idx), query, Core.NORM_L2);
            if (dist < minDist) {
                minId    = idx;
                minDist  = dist;
                minClass = _labels.get(idx,0)[0];
            }
        }
        return new MatOfFloat((float)(minClass),(float)(minDist),(float)(minId));
    }
}



//~ class ClassifierFisher extends ClassifierEigen
//~ {
    //~ public ClassifierFisher(int num_components)  {
        //~ super(num_components);
    //~ }

    //~ int unique( ) {
    //~ }
    //~ public boolean train(Mat data, Mat labels) {
        //~ int N = data.rows;
        //~ int C = unique(labels);
        //~ if((_num_components <= 0) || (_num_components > (C-1))) // btw, why C-1 ?
            //~ _num_components = (C-1);

        //~ // step one, do pca on the original(pixel) data:
        //~ PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, (N-C));
        //~ _mean = pca.mean.reshape(1,1);

        //~ // step two, do lda on data projected to pca space:
        //~ Mat proj = LDA::subspaceProject(pca.eigenvectors.t(), _mean, data);
        //~ LDA lda(proj, labels, _num_components);

        //
        //// ok, there's no LDA yet in java. pity.
        //// nice idea dies here.
        //

        //~ // step three, combine both:
        //~ Mat leigen; 
        //~ lda.eigenvectors().convertTo(leigen, pca.eigenvectors.type());
        //~ gemm(pca.eigenvectors, leigen, 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);

        //~ // step four, project training images to lda space for prediction:
        //~ _labels = labels;
        //~ save_projections(data);
        //~ return 1;
    //~ }
//~ };



