<pre>

 belgium traffic sign classification set from:
   http://btsd.ethz.ch/shareddata/
   ( BelgiumTSC_Training.zip & BelgiumTSC_Testing.zip
     i changed the directory structure, so it looks like:
     BelgiumTSC
     ├───Testing
     │   ├───00000
     │   ├───00001
     ...
     │   └───00061
     └───Training
         ├───00000
         ├───00001
     ...
         └───00061 )
 tiny-dnn code from:
   https://github.com/tiny-dnn/
   (main tries to load a json model)

 also comes with opencv SVM & ANN_MLP reference impl.
    svm accuracy: 0.882399 (tsc -s)
    mlp accuracy: 0.857932 (tsc -m)
    dnn accuracy: 0.945935 (tsc -t -j=tsc32.txt -w=tsc32.net)

new: transfer learning!
    tsv_transfer.cpp will try to abuse https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    (as a glorified feature preprocessor for a 4 stage cv::ANN_MLP)
    transfer accuracy: 0.971575 !
    (takes ages, though, like half a second per image)
