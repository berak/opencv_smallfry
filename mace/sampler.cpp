struct MaceSampler : MACE {
    Ptr<MACE> samp[4];
    int siz;
    MaceSampler(int siz, int salt) : siz(siz) {
        samp[0] = MACE::create(siz,salt); // whole
        samp[1] = MACE::create(siz/2,salt); // top left
        samp[2] = MACE::create(siz/2,salt); // top right
        samp[3] = MACE::create(siz/2,salt); // bot center
    }
    void train(cv::InputArrayOfArrays input) {
        vector<Mat> images;
        input.getMatVector(images);
        samp[0]->compute(images);
        vector<Mat> tl,tr,bc;
        for (size_t i=0; i<images.size(); i++) {
            tl.push_back(Mat(images[i], Rect(0,0,siz/2,siz/2)));
            tr.push_back(Mat(images[i], Rect(siz/2,0,siz/2,siz/2)));
            bc.push_back(Mat(images[i], Rect(siz/4,siz/2,siz/2,siz/2)));
        }
        samp[1]->compute(tl);
        samp[2]->compute(tr);
        samp[3]->compute(bc);
    }
    double correlate(InputArray img) const {
        double r0 = samp[0]->correlate(img);
        double r1 = samp[1]->correlate(Mat(img, Rect(0,0,siz/2,siz/2)));
        double r2 = samp[2]->correlate(Mat(img, Rect(siz/2,0,siz/2,siz/2)));
        double r3 = samp[3]->correlate(Mat(img, Rect(siz/4,siz/2,siz/2,siz/2)));
        return (r0+r1+r2+r3) / 4;
    }
};
