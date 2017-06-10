
Minimum Average Correlation Energy Filter

  * useful for authentification with (cancellable) biometrical features.
  * (does not need many positives to train (~10), and no negatives at all, also robust to noise/salting)
  * algorithm largely taken from : https://github.com/polyu/faceservergpl


![re1](https://github.com/berak/opencv_smallfry/raw/master/mace/RE1.png)  &nbsp;&nbsp;&nbsp;
![re2](https://github.com/berak/opencv_smallfry/raw/master/mace/RE2.png)  &nbsp;&nbsp;&nbsp;
![re3](https://github.com/berak/opencv_smallfry/raw/master/mace/RE3.png)


use it like:

    Ptr<MACE> mace = MACE::create(64);

    vector<Mat> pos_images = ...
    mace->train(pos_images);

    Mat query = ...
    bool same = mace->same(query);


2 samples are supplied,
* att.cpp:  shootout on att faces (you'll need the db for this)
* webcam.cpp: train & test from webcam images
