you need the [opencv_contrib repo](https://github.com/Itseez/opencv_contrib), also, the java bindings are not enabled by default.

add "java" [at the end of line 2 here](https://github.com/Itseez/opencv_contrib/blob/master/modules/face/CMakeLists.txt#L2),

then rerun cmake && make && make install
