PS1=">"

# jikes, we'll haveto build cmake from scratch !
curl -s https://cmake.org/files/v3.8/cmake-3.8.0.tar.Z > cmake.tar.gz
tar zxf cmake.tar.gz
cd cmake-3.8.0/
sh configure
make
#cp bin/cmake ..  ## needs absolute path nowadays
cd ..

# if we want java, we 'll need the ant tool; had to fall back to 1.9.9 due to java on server
curl https://archive.apache.org/dist/ant/binaries/apache-ant-1.9.11-bin.tar.gz > ant.tgz
tar zxf ant.tgz

git clone https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build

echo -n /app/cmake-3.8.0/bin/cmake -G \"Unix Makefiles\" -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DBUILD_TIFF=> cm.sh
echo -n  OFF -DOPENCV_ENABLE_NONFREE=ON -DWITH_TIFF=OFF -DWITH_IPP=OFF -DBUILD_PNG=OFF -DWITH_PNG=>> cm.sh
echo -n OFF -DBUILD_ZLIB=ON -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DBUILD_OPENEXR=OFF -DBUILD_SHARED_LIBS= >> cm.sh
echo -n  OFF -DCMAKE_INSTALL_PREFIX=/app/ocv3 -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=  >> cm.sh
echo  OFF -DBUILD_opencv_apps=OFF -DBUILD_examples=OFF -DANT_EXECUTABLE=../../apache-ant-1.9.11/bin/ant .. >> cm.sh
bash cm.sh

make -j4 install

cd ..
cd ..
tar -czf ocv3.tgz ocv3

# this is where the "manual" part starts ..
ssh-keygen -t rsa -C "px1704@web.de"
cat .ssh/id_rsa.pub

#~ # -> github -> account settings -> ssh keys -> add above key
mkdir sugar
cd sugar
git clone git@github.com:berak/sugarcoatedchili.git  # git/ssh url !!  (yes)

cp ../ocv3.tgz sugarcoatedchili
cd sugarcoatedchili
git config --global user.email px1704@web.de
git config --global user.name berak
git add ocv3.tgz
git commit -a -m "ocv400"
git push
cd ..
cd ..


# viola ;)
