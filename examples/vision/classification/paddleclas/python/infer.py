from pickletools import optimize
import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleClas model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--topk", type=int, default=1, help="Return topk results.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu' or 'ipu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.device.lower() == "ipu":
        option.use_ipu()

    if args.use_trt:
        option.use_trt_backend()

    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.7/dist-packages/fastdeploy/libs/third_libs/paddlelite/lib/
    option.use_lite_backend()
    option.use_cann()
    option.set_lite_nnadapter_device_names(["huawei_ascend_npu"])

    return option


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = build_option(args)

model_file = os.path.join(args.model, "inference.pdmodel")
params_file = os.path.join(args.model, "inference.pdiparams")
config_file = os.path.join(args.model, "inference_cls.yaml")
model = fd.vision.classification.PaddleClasModel(
    model_file, params_file, config_file, runtime_option=runtime_option)

# 预测图片分类结果
im = cv2.imread(args.image)
result = model.predict(im, args.topk)
print(result)
