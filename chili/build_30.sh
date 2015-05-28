PS1=">"

curl -s http://www.cmake.org/files/v2.8/cmake-2.8.12.1.tar.Z > cmake.tar.gz
tar zxf cmake.tar.gz
cd cmake-2.8.12.1/

sh configure
make
cp bin/cmake ..
cd ..

#tar -czf cmake.tgz  cmake-2.8.12.1



curl http://apache.mirror.digionline.de//ant/binaries/apache-ant-1.9.4-bin.tar.gz > ant.tgz
tar zxf ant.tgz

git clone https://github.com/Itseez/opencv_contrib
git clone https://github.com/Itseez/opencv.git
cd opencv
git fetch

#../cmake -G "Unix Makefiles" -DOPENCV_EXTRA_MODULES_PATH=../opencv_extra/modules -DBUILD_TIFF=ON -DWITH_TIFF=ON -DBUILD_PNG=ON -DWITH_PNG=ON -DBUILD_JPEG=ON -DWITH_JPEG=ON -DBUILD_ZLIB=ON -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DBUILD_OPENEXR=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/app/ocv3 -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_apps=OFF -DBUILD_examples=OFF


echo -n ../cmake -G \"Unix Makefiles\" -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -DBUILD_TIFF=> cm.in
echo -n  ON -DWITH_TIFF=ON -DWITH_IPP=OFF -DBUILD_PNG=ON -DWITH_PNG=>> cm.in
echo -n ON -DBUILD_ZLIB=ON -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DBUILD_OPENEXR=OFF -DBUILD_SHARED_LIBS= >> cm.in
echo -n  OFF -DCMAKE_INSTALL_PREFIX=/app/ocv3 -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=  >> cm.in
echo  OFF -DBUILD_opencv_apps=OFF -DBUILD_examples=OFF -DANT_EXECUTABLE=../apache-ant-1.9.4/bin/ant >> cm.in

bash ./cm.in

make 
make install

cd ..
tar -czf ocv3.tgz ocv3

ssh-keygen -t rsa -C "px1704@web.de"
cat .ssh/id_rsa.pub

# -> github -> accout settings -> ssh keys -> add above key

mkdir sugar
cd sugar
git clone git@github.com:berak/sugarcoatedchili.git  # git/ssh url !!
# answer yes

cp ../ocv3.tgz sugarcoatedchili
#cp ../cmake.tgz sugarcoatedchili
cd sugarcoatedchili
git add ocv3.tgz
#git add cmake.tgz
git commit -a -m "ocv3.tgz"
git push
cd ..
cd ..

# don't forget to push the heroku repo, so it gets updated !
# viola ;)
