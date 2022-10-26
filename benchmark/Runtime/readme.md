# 文件说明

- `infer.cc` 测试FP32模型和INT8模型
- `infer_PM.cc` 测试FP32+Pinned mem， 和 INT8 + Pinned mem 数据
- `Infer_fp16.cc` 给NV-TRT 在推理量化模型时, 同时开启FP16模式
- `Infer_fp16_PM.cc` 在NV-TRT上, 测试 INT8+FP16+Pinned mem 数据

#  参考用法


```
1.更改compile.sh中的DFASTDEPLOY_INSTALL_DIR 为自己的FastDeploy路径

bash compile.sh


2. 开始跑用例  ./infer_demo 模型路径 Rumtime调用次数 Runtime选择

./infer_demo model_dir 1000 0   # 使用ORT在CPU上推理1000次,得到平均latency
./infer_demo model_dir 1000 1   # 使用PaddleInference在CPU上推理1000次,得到平均latency
./infer_demo model_dir 1000 2   # 使用NV-TRT在GPU上推理1000次,得到平均latency
./infer_demo model_dir 1000 3   # 使用PP-TRT在GPU上推理1000次,得到平均latency

```
