PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc)
add_executable(infer_demo_fp16 ${PROJECT_SOURCE_DIR}/infer_fp16.cc)
add_executable(infer_demo_fp16_PM ${PROJECT_SOURCE_DIR}/infer_fp16_PM.cc)
add_executable(infer_demo_PM ${PROJECT_SOURCE_DIR}/infer_PM.cc)

# 添加FastDeploy库依赖
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})
target_link_libraries(infer_demo_fp16 ${FASTDEPLOY_LIBS})
target_link_libraries(infer_demo_fp16_PM ${FASTDEPLOY_LIBS})
target_link_libraries(infer_demo_PM ${FASTDEPLOY_LIBS})
