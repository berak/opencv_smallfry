package texture;

import org.opencv.core.*;
import java.util.*;

public class FaceRec {
	Extractor  ext;
	Classifier cls;

	public FaceRec(String extractor, String classifier) {
		ext = Extractor.create(extractor);
		cls = Classifier.create(classifier);
        System.err.println(ext+" "+cls);
	}


	public boolean train(List<Mat> dataset, Mat labels) {
		Mat features = ext.extract(dataset);
		return cls.train(features.reshape(1,labels.rows()),labels);
	}
	public Mat predict(Mat query) {
		Mat features = ext.extract(query);
		return cls.predict(features.reshape(1,1));
	}
}


