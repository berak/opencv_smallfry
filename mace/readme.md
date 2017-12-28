
Minimum Average Correlation Energy Filter

  * useful for authentification with (cancellable) biometrical features.
  * (does not need many positives to train (~10), and no negatives at all, also robust to noise/salting)
  * algorithm largely taken from : https://code.google.com/archive/p/pam-face-authentication/



use it like:

    Ptr<MACE> mace = MACE::create(64);

    vector<Mat> pos_images = ...
    mace->train(pos_images);

    Mat query = ...
    bool same = mace->same(query);


3 samples are supplied,
* att.cpp:  shootout on att faces (you'll need the db for this)
* webcam.cpp: train & test from webcam images
* fingerprints.cpp: train & test on fingerprint images (from the opencv-blueprints book)
