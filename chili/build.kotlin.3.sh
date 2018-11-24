#tar -xf ocv3.tgz
rm ocv.jar
/app/kotlin-1.1.0/bin/kotlinc src/ocv.kt -classpath "/app/ocv3/share/OpenCV/java/opencv-341.jar;ocv.jar" -include-runtime -d ocv.jar
java -jar ocv.jar
