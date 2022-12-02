// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/runtime.h"

#include "fastdeploy/utils/unique_ptr.h"
#include "fastdeploy/utils/utils.h"

#ifdef ENABLE_ORT_BACKEND
#include "fastdeploy/backends/ort/ort_backend.h"
#endif

#ifdef ENABLE_TRT_BACKEND
#include "fastdeploy/backends/tensorrt/trt_backend.h"
#endif

#ifdef ENABLE_PADDLE_BACKEND
#include "fastdeploy/backends/paddle/paddle_backend.h"
#endif

#ifdef ENABLE_POROS_BACKEND
#include "fastdeploy/backends/poros/poros_backend.h"
#endif

#ifdef ENABLE_OPENVINO_BACKEND
#include "fastdeploy/backends/openvino/ov_backend.h"
#endif

#ifdef ENABLE_LITE_BACKEND
#include "fastdeploy/backends/lite/lite_backend.h"
#endif

#ifdef ENABLE_RKNPU2_BACKEND
#include "fastdeploy/backends/rknpu/rknpu2/rknpu2_backend.h"
#endif

namespace fastdeploy {

std::vector<Backend> GetAvailableBackends() {
  std::vector<Backend> backends;
#ifdef ENABLE_ORT_BACKEND
  backends.push_back(Backend::ORT);
#endif
#ifdef ENABLE_TRT_BACKEND
  backends.push_back(Backend::TRT);
#endif
#ifdef ENABLE_PADDLE_BACKEND
  backends.push_back(Backend::PDINFER);
#endif
#ifdef ENABLE_POROS_BACKEND
  backends.push_back(Backend::POROS);
#endif
#ifdef ENABLE_OPENVINO_BACKEND
  backends.push_back(Backend::OPENVINO);
#endif
#ifdef ENABLE_LITE_BACKEND
  backends.push_back(Backend::LITE);
#endif
#ifdef ENABLE_RKNPU2_BACKEND
  backends.push_back(Backend::RKNPU2);
#endif
  return backends;
}

bool IsBackendAvailable(const Backend& backend) {
  std::vector<Backend> backends = GetAvailableBackends();
  for (size_t i = 0; i < backends.size(); ++i) {
    if (backend == backends[i]) {
      return true;
    }
  }
  return false;
}

std::string Str(const Backend& b) {
  if (b == Backend::ORT) {
    return "Backend::ORT";
  } else if (b == Backend::TRT) {
    return "Backend::TRT";
  } else if (b == Backend::PDINFER) {
    return "Backend::PDINFER";
  } else if (b == Backend::POROS) {
    return "Backend::POROS";
  } else if (b == Backend::RKNPU2) {
    return "Backend::RKNPU2";
  }else if (b == Backend::OPENVINO) {
    return "Backend::OPENVINO";
  } else if (b == Backend::LITE) {
    return "Backend::PDLITE";
  }
  return "UNKNOWN-Backend";
}

std::ostream& operator<<(std::ostream& out, const Backend& backend) {
  if (backend == Backend::ORT) {
    out << "Backend::ORT";
  } else if (backend == Backend::TRT) {
    out << "Backend::TRT";
  } else if (backend == Backend::PDINFER) {
    out << "Backend::PDINFER";
  } else if (backend == Backend::OPENVINO) {
    out << "Backend::OPENVINO";
  } else if (backend == Backend::RKNPU2) {
    out << "Backend::RKNPU2";
  }else if (backend == Backend::POROS) {
    out << "Backend::POROS";
  } else if (backend == Backend::LITE) {
    out << "Backend::PDLITE";
  } else {
    out << "UNKNOWN-Backend";
  }
  return out;
}

bool CheckModelFormat(const std::string& model_file,
                      const ModelFormat& model_format) {
  if (model_format == ModelFormat::PADDLE) {
    if (model_file.size() < 8 ||
        model_file.substr(model_file.size() - 8, 8) != ".pdmodel") {
      FDERROR << "With model format of ModelFormat::PADDLE, the model file "
                 "should ends with `.pdmodel`, but now it's "
              << model_file << std::endl;
      return false;
    }
  } else if (model_format == ModelFormat::ONNX) {
    if (model_file.size() < 5 ||
        model_file.substr(model_file.size() - 5, 5) != ".onnx") {
      FDERROR << "With model format of ModelFormat::ONNX, the model file "
                 "should ends with `.onnx`, but now it's "
              << model_file << std::endl;
      return false;
    }
  } else if (model_format == ModelFormat::RKNN) {
    if (model_file.size() < 5 ||
        model_file.substr(model_file.size() - 5, 5) != ".rknn") {
      FDERROR << "With model format of ModelFormat::RKNN, the model file "
                 "should ends with `.rknn`, but now it's "
              << model_file << std::endl;
      return false;
    }
  } else if (model_format == ModelFormat::TORCHSCRIPT) {
    if (model_file.size() < 3 ||
        model_file.substr(model_file.size() - 3, 3) != ".pt") {
      FDERROR << "With model format of ModelFormat::TORCHSCRIPT, the model file "
                 "should ends with `.pt`, but now it's "
              << model_file << std::endl;
      return false;
    }
  } else {
    FDERROR << "Only support model format with frontend ModelFormat::PADDLE / "
               "ModelFormat::ONNX / ModelFormat::RKNN / ModelFormat::TORCHSCRIPT."
            << std::endl;
    return false;
  }
  return true;
}

ModelFormat GuessModelFormat(const std::string& model_file) {
  if (model_file.size() > 8 &&
      model_file.substr(model_file.size() - 8, 8) == ".pdmodel") {
    FDINFO << "Model Format: PaddlePaddle." << std::endl;
    return ModelFormat::PADDLE;
  } else if (model_file.size() > 5 &&
             model_file.substr(model_file.size() - 5, 5) == ".onnx") {
    FDINFO << "Model Format: ONNX." << std::endl;
    return ModelFormat::ONNX;
  } else if (model_file.size() > 3 &&
             model_file.substr(model_file.size() - 3, 3) == ".pt") {
    FDINFO << "Model Format: Torchscript." << std::endl;
    return ModelFormat::TORCHSCRIPT;
  } else if (model_file.size() > 5 &&
             model_file.substr(model_file.size() - 5, 5) == ".rknn") {
    FDINFO << "Model Format: RKNN." << std::endl;
    return ModelFormat::RKNN;
  }

  FDERROR << "Cannot guess which model format you are using, please set "
             "RuntimeOption::model_format manually."
          << std::endl;
  return ModelFormat::PADDLE;
}

void RuntimeOption::SetModelPath(const std::string& model_path,
                                 const std::string& params_path,
                                 const ModelFormat& format) {
  if (format == ModelFormat::PADDLE) {
    model_file = model_path;
    params_file = params_path;
    model_format = ModelFormat::PADDLE;
  } else if (format == ModelFormat::ONNX) {
    model_file = model_path;
    model_format = ModelFormat::ONNX;
  } else if (format == ModelFormat::TORCHSCRIPT) {
    model_file = model_path;
    model_format = ModelFormat::TORCHSCRIPT;
  } else {
    FDASSERT(
        false,
        "The model format only can be ModelFormat::PADDLE/ModelFormat::ONNX/ModelFormat::TORCHSCRIPT.");
  }
}

void RuntimeOption::UseGpu(int gpu_id) {
#ifdef WITH_GPU
  device = Device::GPU;
  device_id = gpu_id;
#else
  FDWARNING << "The FastDeploy didn't compile with GPU, will force to use CPU."
            << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::UseCpu() { device = Device::CPU; }

void RuntimeOption::UseRKNPU2(fastdeploy::rknpu2::CpuName rknpu2_name,
                              fastdeploy::rknpu2::CoreMask rknpu2_core) {
  rknpu2_cpu_name_ = rknpu2_name;
  rknpu2_core_mask_ = rknpu2_core;
  device = Device::RKNPU;
}

void RuntimeOption::UseTimVX() {
  enable_timvx = true;
  device = Device::TIMVX;
  UseLiteBackend();
}

void RuntimeOption::UseCANN(){
  enable_cann = true;
  device = Device::CANN;
  UseLiteBackend();
}

void RuntimeOption::SetExternalStream(void* external_stream) {
  external_stream_ = external_stream;
}

void RuntimeOption::SetCpuThreadNum(int thread_num) {
  FDASSERT(thread_num > 0, "The thread_num must be greater than 0.");
  cpu_thread_num = thread_num;
}

void RuntimeOption::SetOrtGraphOptLevel(int level) {
  std::vector<int> supported_level{-1, 0, 1, 2};
  auto valid_level = std::find(supported_level.begin(), supported_level.end(),
                               level) != supported_level.end();
  FDASSERT(valid_level, "The level must be -1, 0, 1, 2.");
  ort_graph_opt_level = level;
}

// use paddle inference backend
void RuntimeOption::UsePaddleBackend() {
#ifdef ENABLE_PADDLE_BACKEND
  backend = Backend::PDINFER;
#else
  FDASSERT(false, "The FastDeploy didn't compile with Paddle Inference.");
#endif
}

// use onnxruntime backend
void RuntimeOption::UseOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  backend = Backend::ORT;
#else
  FDASSERT(false, "The FastDeploy didn't compile with OrtBackend.");
#endif
}

// use poros backend
void RuntimeOption::UsePorosBackend() {
#ifdef ENABLE_POROS_BACKEND
  backend = Backend::POROS;
#else
  FDASSERT(false, "The FastDeploy didn't compile with PorosBackend.");
#endif
}

void RuntimeOption::UseTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  backend = Backend::TRT;
#else
  FDASSERT(false, "The FastDeploy didn't compile with TrtBackend.");
#endif
}

void RuntimeOption::UseOpenVINOBackend() {
#ifdef ENABLE_OPENVINO_BACKEND
  backend = Backend::OPENVINO;
#else
  FDASSERT(false, "The FastDeploy didn't compile with OpenVINO.");
#endif
}

void RuntimeOption::UseLiteBackend() {
#ifdef ENABLE_LITE_BACKEND
  backend = Backend::LITE;
#else
  FDASSERT(false, "The FastDeploy didn't compile with Paddle Lite.");
#endif
}

void RuntimeOption::SetPaddleMKLDNN(bool pd_mkldnn) {
  pd_enable_mkldnn = pd_mkldnn;
}

void RuntimeOption::DeletePaddleBackendPass(const std::string& pass_name) {
  pd_delete_pass_names.push_back(pass_name);
}
void RuntimeOption::EnablePaddleLogInfo() { pd_enable_log_info = true; }

void RuntimeOption::DisablePaddleLogInfo() { pd_enable_log_info = false; }

void RuntimeOption::EnablePaddleToTrt() {
  FDASSERT(backend == Backend::TRT, "Should call UseTrtBackend() before call EnablePaddleToTrt().");
#ifdef ENABLE_PADDLE_BACKEND
  FDINFO << "While using TrtBackend with EnablePaddleToTrt, FastDeploy will change to use Paddle Inference Backend." << std::endl;
  backend = Backend::PDINFER;
  pd_enable_trt = true;
#else
  FDASSERT(false, "While using TrtBackend with EnablePaddleToTrt, require the FastDeploy is compiled with Paddle Inference Backend, please rebuild your FastDeploy.");
#endif
}

void RuntimeOption::SetPaddleMKLDNNCacheSize(int size) {
  FDASSERT(size > 0, "Parameter size must greater than 0.");
  pd_mkldnn_cache_size = size;
}

void RuntimeOption::SetOpenVINODevice(const std::string& name) {
  openvino_device = name;
}

void RuntimeOption::EnableLiteFP16() {
  lite_enable_fp16 = true;
}

void RuntimeOption::DisableLiteFP16() {
  lite_enable_fp16 = false;
}
void RuntimeOption::EnableLiteInt8() {
  lite_enable_int8 = true;
}

void RuntimeOption::DisableLiteInt8() {
  lite_enable_int8 = false;
}
void RuntimeOption::SetLitePowerMode(LitePowerMode mode) {
  lite_power_mode = mode;
}

void RuntimeOption::SetLiteOptimizedModelDir(
    const std::string& optimized_model_dir) {
  lite_optimized_model_dir = optimized_model_dir;
}

void RuntimeOption::EnableCANN() {
  enable_cann = true;
}

void RuntimeOption::SetLiteNNAdapterSubgraphPartitionPath(
    const std::string& nnadapter_subgraph_partition_config_path) {
  lite_nnadapter_subgraph_partition_config_path = nnadapter_subgraph_partition_config_path;
}

void RuntimeOption::SetLiteNNAdapterSubgraphPartitionConfigBuffer(
      const std::string& nnadapter_subgraph_partition_config_buffer){
  lite_nnadapter_subgraph_partition_config_buffer = nnadapter_subgraph_partition_config_buffer;
}

void RuntimeOption::SetLiteNNAdapterDeviceNames(const std::vector<std::string>& nnadapter_device_names){
  lite_nnadapter_device_names = nnadapter_device_names; 
}

void RuntimeOption::SetLiteNNAdapterContextProperties(const std::string& nnadapter_context_properties){
  lite_nnadapter_context_properties = nnadapter_context_properties; 
}

void RuntimeOption::SetLiteNNAdapterModelCacheDir(const std::string& nnadapter_model_cache_dir){
  lite_nnadapter_model_cache_dir = nnadapter_model_cache_dir;
}

void RuntimeOption::SetLiteNNAdapterModelCacheBuffers(
      const std::string& nnadapter_model_cache_token,
      const std::vector<char>& nnadapter_model_cache_buffer){
  lite_nnadapter_model_cache_token = nnadapter_model_cache_token; 
  lite_nnadapter_model_cache_buffer = nnadapter_model_cache_buffer;
}

void RuntimeOption::SetLiteNNAdapterDynamicShapeInfo(
      const std::map<std::string, std::vector<std::vector<int64_t>>>&
          nnadapter_dynamic_shape_info){
  lite_nnadapter_dynamic_shape_info = nnadapter_dynamic_shape_info; 
}

void RuntimeOption::SetLiteNNAdapterMixedPrecisionQuantizationConfigPath(
      const std::string& nnadapter_mixed_precision_quantization_config_path){
        lite_nnadapter_mixed_precision_quantization_config_path = nnadapter_mixed_precision_quantization_config_path;
}

void RuntimeOption::SetLiteNNAdapterMixedPrecisionQuantizationConfigBuffer(
      const std::string& nnadapter_mixed_precision_quantization_config_buffer){
        lite_nnadapter_mixed_precision_quantization_config_buffer = nnadapter_mixed_precision_quantization_config_buffer;
}

void RuntimeOption::SetTrtInputShape(const std::string& input_name,
                                     const std::vector<int32_t>& min_shape,
                                     const std::vector<int32_t>& opt_shape,
                                     const std::vector<int32_t>& max_shape) {
  trt_min_shape[input_name].clear();
  trt_max_shape[input_name].clear();
  trt_opt_shape[input_name].clear();
  trt_min_shape[input_name].assign(min_shape.begin(), min_shape.end());
  if (opt_shape.size() == 0) {
    trt_opt_shape[input_name].assign(min_shape.begin(), min_shape.end());
  } else {
    trt_opt_shape[input_name].assign(opt_shape.begin(), opt_shape.end());
  }
  if (max_shape.size() == 0) {
    trt_max_shape[input_name].assign(min_shape.begin(), min_shape.end());
  } else {
    trt_max_shape[input_name].assign(max_shape.begin(), max_shape.end());
  }
}

void RuntimeOption::SetTrtMaxWorkspaceSize(size_t max_workspace_size) {
  trt_max_workspace_size = max_workspace_size;
}
void RuntimeOption::SetTrtMaxBatchSize(size_t max_batch_size){
  trt_max_batch_size = max_batch_size; 
}

void RuntimeOption::EnableTrtFP16() { trt_enable_fp16 = true; }

void RuntimeOption::DisableTrtFP16() { trt_enable_fp16 = false; }

void RuntimeOption::EnablePinnedMemory() { enable_pinned_memory = true; }

void RuntimeOption::DisablePinnedMemory() { enable_pinned_memory = false; }

void RuntimeOption::SetTrtCacheFile(const std::string& cache_file_path) {
  trt_serialize_file = cache_file_path;
}

void RuntimeOption::SetOpenVINOStreams(int num_streams) {
  ov_num_streams = num_streams;
}

bool Runtime::Compile(std::vector<std::vector<FDTensor>>& prewarm_tensors,
                      const RuntimeOption& _option) {
#ifdef ENABLE_POROS_BACKEND
  option = _option;
  auto poros_option = PorosBackendOption();
  poros_option.use_gpu = (option.device == Device::GPU) ? true : false;
  poros_option.gpu_id = option.device_id;
  poros_option.long_to_int = option.long_to_int;
  poros_option.use_nvidia_tf32 = option.use_nvidia_tf32;
  poros_option.unconst_ops_thres = option.unconst_ops_thres;
  poros_option.poros_file = option.poros_file;
  poros_option.is_dynamic = option.is_dynamic;
  poros_option.enable_fp16 = option.trt_enable_fp16;
  poros_option.max_batch_size = option.trt_max_batch_size;
  poros_option.max_workspace_size = option.trt_max_workspace_size;
  FDASSERT(option.model_format == ModelFormat::TORCHSCRIPT,
           "PorosBackend only support model format of ModelFormat::TORCHSCRIPT.");
  backend_ = utils::make_unique<PorosBackend>();
  auto casted_backend = dynamic_cast<PorosBackend*>(backend_.get());
  FDASSERT(
      casted_backend->Compile(option.model_file, prewarm_tensors, poros_option),
      "Load model from Torchscript failed while initliazing PorosBackend.");
#else
  FDASSERT(false,
           "PorosBackend is not available, please compiled with "
           "ENABLE_POROS_BACKEND=ON.");
#endif
  return true;
}

void RuntimeOption::EnablePaddleTrtCollectShape() {
  pd_collect_shape = true;
}

void RuntimeOption::DisablePaddleTrtCollectShape() {
  pd_collect_shape = false;
}

void RuntimeOption::UseIpu(int device_num, int micro_batch_size,
                           bool enable_pipelining, int batches_per_step) {
#ifdef WITH_IPU
  device = Device::IPU;
  ipu_device_num = device_num;
  ipu_micro_batch_size = micro_batch_size;
  ipu_enable_pipelining = enable_pipelining;
  ipu_batches_per_step = batches_per_step;
#else
  FDWARNING << "The FastDeploy didn't compile with IPU, will force to use CPU."
            << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::SetIpuConfig(bool enable_fp16, int replica_num,
                                 float available_memory_proportion,
                                 bool enable_half_partial) {
  ipu_enable_fp16 = enable_fp16;
  ipu_replica_num = replica_num;
  ipu_available_memory_proportion = available_memory_proportion;
  ipu_enable_half_partial = enable_half_partial;
}

bool Runtime::Init(const RuntimeOption& _option) {
  option = _option;
  if (option.model_format == ModelFormat::AUTOREC) {
    option.model_format = GuessModelFormat(_option.model_file);
  }
  if (option.backend == Backend::UNKNOWN) {
    if (IsBackendAvailable(Backend::ORT)) {
      option.backend = Backend::ORT;
    } else if (IsBackendAvailable(Backend::PDINFER)) {
      option.backend = Backend::PDINFER;
    } else if (IsBackendAvailable(Backend::POROS)) {
      option.backend = Backend::POROS;
    } else if (IsBackendAvailable(Backend::OPENVINO)) {
      option.backend = Backend::OPENVINO;
    } else if (IsBackendAvailable(Backend::RKNPU2)) {
      option.backend = Backend::RKNPU2;
    } else {
      FDERROR << "Please define backend in RuntimeOption, current it's "
                 "Backend::UNKNOWN."
              << std::endl;
      return false;
    }
  }

  if (option.backend == Backend::ORT) {
    FDASSERT(option.device == Device::CPU || option.device == Device::GPU,
             "Backend::ORT only supports Device::CPU/Device::GPU.");
    CreateOrtBackend();
    FDINFO << "Runtime initialized with Backend::ORT in " << Str(option.device)
           << "." << std::endl;
  } else if (option.backend == Backend::TRT) {
    FDASSERT(option.device == Device::GPU,
             "Backend::TRT only supports Device::GPU.");
    CreateTrtBackend();
    FDINFO << "Runtime initialized with Backend::TRT in " << Str(option.device)
           << "." << std::endl;
  } else if (option.backend == Backend::PDINFER) {
    FDASSERT(
        option.device == Device::CPU || option.device == Device::GPU ||
            option.device == Device::IPU,
        "Backend::PDINFER only supports Device::CPU/Device::GPU/Device::IPU.");
    FDASSERT(
        option.model_format == ModelFormat::PADDLE,
        "Backend::PDINFER only supports model format of ModelFormat::PADDLE.");
    CreatePaddleBackend();
    FDINFO << "Runtime initialized with Backend::PDINFER in "
           << Str(option.device) << "." << std::endl;
  } else if (option.backend == Backend::POROS) {
    FDASSERT(option.device == Device::CPU || option.device == Device::GPU,
             "Backend::POROS only supports Device::CPU/Device::GPU.");
    FDASSERT(
        option.model_format == ModelFormat::TORCHSCRIPT,
        "Backend::POROS only supports model format of ModelFormat::TORCHSCRIPT.");
    FDINFO << "Runtime initialized with Backend::POROS in "
           << Str(option.device) << "." << std::endl;
    return true;
  } else if (option.backend == Backend::OPENVINO) {
    FDASSERT(option.device == Device::CPU,
             "Backend::OPENVINO only supports Device::CPU");
    CreateOpenVINOBackend();
    FDINFO << "Runtime initialized with Backend::OPENVINO in "
           << Str(option.device) << "." << std::endl;
  } else if (option.backend == Backend::LITE) {
    FDASSERT(option.device == Device::CPU || option.device == Device::TIMVX || option.device == Device::CANN,
             "Backend::LITE only supports Device::CPU/Device::TIMVX/Device::CANN.");
    CreateLiteBackend();
    FDINFO << "Runtime initialized with Backend::LITE in " << Str(option.device)
           << "." << std::endl;
  } else if (option.backend == Backend::RKNPU2) {
    FDASSERT(option.device == Device::RKNPU,
             "Backend::RKNPU2 only supports Device::RKNPU2");
    CreateRKNPU2Backend();

    FDINFO << "Runtime initialized with Backend::RKNPU2 in "
           << Str(option.device) << "." << std::endl;
  } else {
    FDERROR << "Runtime only support "
               "Backend::ORT/Backend::TRT/Backend::PDINFER/Backend::POROS as "
               "backend now."
            << std::endl;
    return false;
  }
  return true;
}

TensorInfo Runtime::GetInputInfo(int index) {
  return backend_->GetInputInfo(index);
}

TensorInfo Runtime::GetOutputInfo(int index) {
  return backend_->GetOutputInfo(index);
}

std::vector<TensorInfo> Runtime::GetInputInfos() {
  return backend_->GetInputInfos();
}

std::vector<TensorInfo> Runtime::GetOutputInfos() {
  return backend_->GetOutputInfos();
}

bool Runtime::Infer(std::vector<FDTensor>& input_tensors,
                    std::vector<FDTensor>* output_tensors) {
  for (auto& tensor: input_tensors) {
    FDASSERT(tensor.device_id < 0 || tensor.device_id == option.device_id,
             "Device id of input tensor(%d) and runtime(%d) are not same.",
             tensor.device_id, option.device_id);
  }
  return backend_->Infer(input_tensors, output_tensors);
}

bool Runtime::Infer() {
  return backend_->Infer(input_tensors_, &output_tensors_, false);
}

void Runtime::BindInputTensor(const std::string& name, FDTensor& input) {
  bool is_exist = false;
  for (auto& t : input_tensors_) {
    if (t.name == name) {
      is_exist = true;
      t.SetExternalData(input.shape, input.dtype,
                        input.MutableData(), input.device,
                        input.device_id);
      break;
    }
  }
  if(!is_exist) {
    FDTensor new_tensor(name);
    new_tensor.SetExternalData(input.shape, input.dtype,
                        input.MutableData(), input.device,
                        input.device_id);
    input_tensors_.emplace_back(std::move(new_tensor));
  }
}

FDTensor* Runtime::GetOutputTensor(const std::string& name) {
  for (auto& t : output_tensors_) {
    if (t.name == name) {
      return &t;
    }
  }
  FDWARNING << "The output name [" << name << "] don't exist." << std::endl;
  return nullptr;
}

void Runtime::CreatePaddleBackend() {
#ifdef ENABLE_PADDLE_BACKEND
  auto pd_option = PaddleBackendOption();
  pd_option.model_file = option.model_file;
  pd_option.params_file = option.params_file;
  pd_option.enable_mkldnn = option.pd_enable_mkldnn;
  pd_option.enable_log_info = option.pd_enable_log_info;
  pd_option.mkldnn_cache_size = option.pd_mkldnn_cache_size;
  pd_option.use_gpu = (option.device == Device::GPU) ? true : false;
  pd_option.use_ipu = (option.device == Device::IPU) ? true : false;
  pd_option.gpu_id = option.device_id;
  pd_option.delete_pass_names = option.pd_delete_pass_names;
  pd_option.cpu_thread_num = option.cpu_thread_num;
  pd_option.enable_pinned_memory = option.enable_pinned_memory;
  pd_option.external_stream_ = option.external_stream_;
#ifdef ENABLE_TRT_BACKEND
  if (pd_option.use_gpu && option.pd_enable_trt) {
    pd_option.enable_trt = true;
    pd_option.collect_shape = option.pd_collect_shape;
    auto trt_option = TrtBackendOption();
    trt_option.gpu_id = option.device_id;
    trt_option.enable_fp16 = option.trt_enable_fp16;
    trt_option.max_batch_size = option.trt_max_batch_size;
    trt_option.max_workspace_size = option.trt_max_workspace_size;
    trt_option.max_shape = option.trt_max_shape;
    trt_option.min_shape = option.trt_min_shape;
    trt_option.opt_shape = option.trt_opt_shape;
    trt_option.serialize_file = option.trt_serialize_file;
    trt_option.enable_pinned_memory = option.enable_pinned_memory;
    pd_option.trt_option = trt_option;
  }
#endif
#ifdef WITH_IPU
  if (pd_option.use_ipu) {
    auto ipu_option = IpuOption();
    ipu_option.ipu_device_num = option.ipu_device_num;
    ipu_option.ipu_micro_batch_size = option.ipu_micro_batch_size;
    ipu_option.ipu_enable_pipelining = option.ipu_enable_pipelining;
    ipu_option.ipu_batches_per_step = option.ipu_batches_per_step;
    ipu_option.ipu_enable_fp16 = option.ipu_enable_fp16;
    ipu_option.ipu_replica_num = option.ipu_replica_num;
    ipu_option.ipu_available_memory_proportion =
        option.ipu_available_memory_proportion;
    ipu_option.ipu_enable_half_partial = option.ipu_enable_half_partial;
    pd_option.ipu_option = ipu_option;
  }
#endif
  FDASSERT(option.model_format == ModelFormat::PADDLE,
           "PaddleBackend only support model format of ModelFormat::PADDLE.");
  backend_ = utils::make_unique<PaddleBackend>();
  auto casted_backend = dynamic_cast<PaddleBackend*>(backend_.get());
  FDASSERT(casted_backend->InitFromPaddle(option.model_file, option.params_file,
                                          pd_option),
           "Load model from Paddle failed while initliazing PaddleBackend.");
#else
  FDASSERT(false,
           "PaddleBackend is not available, please compiled with "
           "ENABLE_PADDLE_BACKEND=ON.");
#endif
}

void Runtime::CreateOpenVINOBackend() {
#ifdef ENABLE_OPENVINO_BACKEND
  auto ov_option = OpenVINOBackendOption();
  ov_option.cpu_thread_num = option.cpu_thread_num;
  ov_option.device = option.openvino_device;
  ov_option.shape_infos = option.ov_shape_infos;
  ov_option.num_streams = option.ov_num_streams;
  for (const auto& op : option.ov_cpu_operators) {
    ov_option.cpu_operators.insert(op);
  }
  FDASSERT(option.model_format == ModelFormat::PADDLE ||
               option.model_format == ModelFormat::ONNX,
           "OpenVINOBackend only support model format of ModelFormat::PADDLE / "
           "ModelFormat::ONNX.");
  backend_ = utils::make_unique<OpenVINOBackend>();
  auto casted_backend = dynamic_cast<OpenVINOBackend*>(backend_.get());

  if (option.model_format == ModelFormat::ONNX) {
    FDASSERT(casted_backend->InitFromOnnx(option.model_file, ov_option),
             "Load model from ONNX failed while initliazing OrtBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, ov_option),
             "Load model from Paddle failed while initliazing OrtBackend.");
  }
#else
  FDASSERT(false,
           "OpenVINOBackend is not available, please compiled with "
           "ENABLE_OPENVINO_BACKEND=ON.");
#endif
}

void Runtime::CreateOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  auto ort_option = OrtBackendOption();
  ort_option.graph_optimization_level = option.ort_graph_opt_level;
  ort_option.intra_op_num_threads = option.cpu_thread_num;
  ort_option.inter_op_num_threads = option.ort_inter_op_num_threads;
  ort_option.execution_mode = option.ort_execution_mode;
  ort_option.use_gpu = (option.device == Device::GPU) ? true : false;
  ort_option.gpu_id = option.device_id;
  ort_option.external_stream_ = option.external_stream_;

  FDASSERT(option.model_format == ModelFormat::PADDLE ||
               option.model_format == ModelFormat::ONNX,
           "OrtBackend only support model format of ModelFormat::PADDLE / "
           "ModelFormat::ONNX.");
  backend_ = utils::make_unique<OrtBackend>();
  auto casted_backend = dynamic_cast<OrtBackend*>(backend_.get());
  if (option.model_format == ModelFormat::ONNX) {
    FDASSERT(casted_backend->InitFromOnnx(option.model_file, ort_option),
             "Load model from ONNX failed while initliazing OrtBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, ort_option),
             "Load model from Paddle failed while initliazing OrtBackend.");
  }
#else
  FDASSERT(false,
           "OrtBackend is not available, please compiled with "
           "ENABLE_ORT_BACKEND=ON.");
#endif
}

void Runtime::CreateTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  auto trt_option = TrtBackendOption();
  trt_option.model_file = option.model_file;
  trt_option.params_file = option.params_file;
  trt_option.model_format = option.model_format;
  trt_option.gpu_id = option.device_id;
  trt_option.enable_fp16 = option.trt_enable_fp16;
  trt_option.enable_int8 = option.trt_enable_int8;
  trt_option.max_batch_size = option.trt_max_batch_size;
  trt_option.max_workspace_size = option.trt_max_workspace_size;
  trt_option.max_shape = option.trt_max_shape;
  trt_option.min_shape = option.trt_min_shape;
  trt_option.opt_shape = option.trt_opt_shape;
  trt_option.serialize_file = option.trt_serialize_file;
  trt_option.enable_pinned_memory = option.enable_pinned_memory;
  trt_option.external_stream_ = option.external_stream_;

  FDASSERT(option.model_format == ModelFormat::PADDLE ||
               option.model_format == ModelFormat::ONNX,
           "TrtBackend only support model format of ModelFormat::PADDLE / "
           "ModelFormat::ONNX.");
  backend_ = utils::make_unique<TrtBackend>();
  auto casted_backend = dynamic_cast<TrtBackend*>(backend_.get());
  if (option.model_format == ModelFormat::ONNX) {
    FDASSERT(casted_backend->InitFromOnnx(option.model_file, trt_option),
             "Load model from ONNX failed while initliazing TrtBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, trt_option),
             "Load model from Paddle failed while initliazing TrtBackend.");
  }
#else
  FDASSERT(false,
           "TrtBackend is not available, please compiled with "
           "ENABLE_TRT_BACKEND=ON.");
#endif
}

void Runtime::CreateLiteBackend() {
#ifdef ENABLE_LITE_BACKEND
  auto lite_option = LiteBackendOption();
  lite_option.threads = option.cpu_thread_num;
  lite_option.enable_int8 = option.lite_enable_int8;
  lite_option.enable_fp16 = option.lite_enable_fp16;
  lite_option.power_mode = static_cast<int>(option.lite_power_mode);
  lite_option.optimized_model_dir = option.lite_optimized_model_dir;
  lite_option.nnadapter_subgraph_partition_config_path = option.lite_nnadapter_subgraph_partition_config_path;
  lite_option.nnadapter_subgraph_partition_config_buffer = option.lite_nnadapter_subgraph_partition_config_buffer;
  lite_option.nnadapter_device_names = option.lite_nnadapter_device_names;
  lite_option.nnadapter_context_properties = option.lite_nnadapter_context_properties;
  lite_option.nnadapter_model_cache_dir = option.lite_nnadapter_model_cache_dir;
  lite_option.nnadapter_dynamic_shape_info = option.lite_nnadapter_dynamic_shape_info;
  lite_option.nnadapter_mixed_precision_quantization_config_path = option.lite_nnadapter_mixed_precision_quantization_config_path;
  lite_option.nnadapter_mixed_precision_quantization_config_buffer = option.lite_nnadapter_mixed_precision_quantization_config_buffer;
  lite_option.enable_timvx = option.enable_timvx;
  lite_option.enable_cann = option.enable_cann;
  FDASSERT(option.model_format == ModelFormat::PADDLE,
           "LiteBackend only support model format of ModelFormat::PADDLE");
  backend_ = utils::make_unique<LiteBackend>();
  auto casted_backend = dynamic_cast<LiteBackend*>(backend_.get());
  FDASSERT(casted_backend->InitFromPaddle(option.model_file, option.params_file,
                                          lite_option),
           "Load model from nb file failed while initializing LiteBackend.");
#else
  FDASSERT(false,
           "LiteBackend is not available, please compiled with "
           "ENABLE_LITE_BACKEND=ON.");
#endif
}

void Runtime::CreateRKNPU2Backend() {
#ifdef ENABLE_RKNPU2_BACKEND
  auto rknpu2_option = RKNPU2BackendOption();
  rknpu2_option.cpu_name = option.rknpu2_cpu_name_;
  rknpu2_option.core_mask = option.rknpu2_core_mask_;
  FDASSERT(option.model_format == ModelFormat::RKNN,
           "RKNPU2Backend only support model format of ModelFormat::RKNN");
  backend_ = utils::make_unique<RKNPU2Backend>();
  auto casted_backend = dynamic_cast<RKNPU2Backend*>(backend_.get());
  FDASSERT(casted_backend->InitFromRKNN(option.model_file, rknpu2_option),
           "Load model from nb file failed while initializing LiteBackend.");
#else
  FDASSERT(false, "RKNPU2Backend is not available, please compiled with "
                  "ENABLE_RKNPU2_BACKEND=ON.");
#endif
}

Runtime* Runtime::Clone(void* stream, int device_id) {
  Runtime* runtime = new Runtime();
  if (option.backend != Backend::OPENVINO
      && option.backend != Backend::PDINFER
      && option.backend != Backend::TRT
      ) {
    runtime->Init(option);
    FDWARNING << "Only OpenVINO/Paddle Inference/TensorRT support \
                  clone engine to  reduce CPU/GPU memory usage now. For "
              << option.backend
              << ", FastDeploy will create a new engine which \
                  will not share memory  with the current runtime."
              << std::endl;
    return runtime;
  }
  FDINFO << "Runtime Clone with Backend:: " << Str(option.backend) << " in " << Str(option.device)
         << "." << std::endl;
  runtime->option = option;
  runtime->backend_ = backend_->Clone(stream, device_id);
  return runtime;
}

}  // namespace fastdeploy
