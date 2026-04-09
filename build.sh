# 构建release使用，会自动将二进制放在release/bin，将依赖放在release/lib
mkdir build
mkdir release
cd build
cmake -DCMAKE_PREFIX_PATH=/usr/include/boost/ -DBOOST_ROOT=/usr/include/boost/ -DCMAKE_INSTALL_PREFIX=../release ..
make -j && make install
cd ..
strip -s release/lib/libIOStream.so  release/lib/libassemble.so  release/lib/libbase.so  release/lib/libgenotype.so  release/lib/libhaplotypecaller.so  release/lib/liblogger.so  release/lib/libpairhmm.so  release/lib/libregion.so  release/lib/libutils.so  release/lib/libwriter.so release/lib/libbqsr.so release/lib/libmain.so release/lib/rovaca