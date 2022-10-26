rm -rf build
mkdir build

cd build

#换成自己的FastDeploy目录
cmake .. -DFASTDEPLOY_INSTALL_DIR=/xieyunyao/project/FastDeploy

make -j
