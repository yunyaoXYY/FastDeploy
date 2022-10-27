#include "fastdeploy/vision.h"

const char sep = '/';

int main(int argc, char* argv[]) {


    // Params
    std::string model_dir = argv[1]; 
    int perf_repeat_time = std::atoi(argv[2]); 
    
    // Set RunTime Option
    fastdeploy::RuntimeOption option;
    int flag = std::atoi(argv[3]);
    if (flag == 0) {
        option.UseCpu();
        option.SetCpuThreadNum(1);
        option.UseOrtBackend();
    }else if (flag == 1) {
        option.UseCpu();
        option.SetCpuThreadNum(1); 
        option.UsePaddleBackend();
    }else if (flag == 2) {
        //NV-TRT
        option.UseGpu();
        option.UseTrtBackend();
        option.EnableTrtFP16();
        option.EnablePinnedMemory();
        option.SetTrtCacheFile(model_dir + sep + "model.trt");
    } else if (flag == 3) {
        //PP-TRT
        //PP-TRT默认即开启了FP16 
        option.UseGpu();
        option.UseTrtBackend();
        option.EnablePinnedMemory();
        option.EnablePaddleToTrt();
    }

    // Init Runtime
    auto model_file = model_dir + sep + "model.pdmodel";
    auto params_file = model_dir + sep + "model.pdiparams";
    option.SetModelPath(model_file,params_file,fastdeploy::ModelFormat::PADDLE); 
    fastdeploy::Runtime run_time;
    run_time.Init(option);
    
    // Get info from Runtime
    auto input_info = run_time.GetInputInfos();
    for (int i = 0 ; i < input_info.size(); ++i){
        std::cout<<"Print the input info from Runtime"<<std::endl;
        std::cout<<input_info[i]<<std::endl;
    }
    
    // Prepare Input Name & Data
    // TODO: multiple inputs
    std::vector<fastdeploy::FDTensor> fake_input_tensors(1); 
    fake_input_tensors[0].name = input_info[0].name; 

    // Fake Input, get shape from Runtime. 
    // TODO: Model with dynamic shape
    auto Height = input_info[0].shape[2];
    auto Width = input_info[0].shape[3]; 
    std::vector<float> input_data(1*3*Height*Width,float(1.0));

    fake_input_tensors[0].SetExternalData({1,3,Height,Width},fastdeploy::FDDataType::FP32,input_data.data(),fastdeploy::Device::CPU);
    std::vector<fastdeploy::FDTensor> output_tensors;
    
    // Warm UP
    int warm_up_times = 100;
    for(int i = 0 ; i < warm_up_times; ++i){
        if (!run_time.Infer(fake_input_tensors, &output_tensors)) {
           fastdeploy::FDERROR << "Failed to inference." << std::endl;
           return false;
       }
    }
    std::cout<<"Finish Warm up"<<std::endl;

    // Perf
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    for(int i = 0 ; i < perf_repeat_time; ++i){
        if (!run_time.Infer(fake_input_tensors, &output_tensors)) {
           fastdeploy::FDERROR << "Failed to inference." << std::endl;
           return false;
       }
    } 

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "You Infer the runtime "<<perf_repeat_time << " times"<<std::endl; 
    std::cout << "The Total time costed is : " << elapsed_seconds.count() << "s\n";
    std::cout << "The avg time is : " << elapsed_seconds.count()/perf_repeat_time << "s\n";

    return 0;
}

