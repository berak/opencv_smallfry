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
 dnn code from:
   https://github.com/tiny-dnn/
   (main tries to load a json model)
 also comes with opencv SVM & ANN_MLP reference impl.
    svm accuracy: 0.882399
    mlp accuracy: 0.857932
    dnn accuracy: 0.945935 (tsc -t -j=tsc32.txt -w=tsc32.net)

</pre>
