PS1=">"

curl -s https://cmake.org/files/v2.8/cmake-2.8.12.1.tar.Z > cmake.tar.gz
tar zxf cmake.tar.gz
cd cmake-2.8.12.1/

sh configure
make
cp bin/cmake ..
cd ..

curl http://apache.mirror.digionline.de//ant/binaries/apache-ant-1.9.4-bin.tar.gz > ant.tgz
tar zxf ant.tgz

git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 2.4
git fetch

echo -n ../cmake -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_examples= > cm.sh
echo -n OFF -DWITH_OPENEXR=OFF -DBUILD_OPENEXR=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX= >> cm.sh
echo -n /app/ocv -DBUILD_ZLIB=ON -DWITH_JASPER=>> cm.sh
echo -n OFF -G \"Unix Makefiles\" -DANT_EXECUTABLE=../apache-ant-1.9.4/bin/ant >> cm.sh
bash cm.sh

make
make install

cd ..
tar -czf ocv.tgz ocv

ssh-keygen -t rsa -C "px1704@web.de"
cat .ssh/id_rsa.pub

#~ # -> github -> accout settings -> ssh keys -> add above key
mkdir sugar
cd sugar
git clone git@github.com:berak/sugarcoatedchili.git  # git/ssh url !!
cp ../ocv.tgz sugarcoatedchili
cd sugarcoatedchili
git add ocv.tgz
git commit -a -m "ocv.tgz"
git push
cd ..
cd ..


# viola ;)
