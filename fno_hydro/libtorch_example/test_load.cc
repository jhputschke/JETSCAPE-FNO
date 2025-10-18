// https://pytorch.org/tutorials/advanced/cpp_export.html
// issues with moving FNO model to CPU in C++, but also in Python ... Follow up!!!!
// Detour: Save GPU/MPS trainied FNO via only weights state_dict. Read in with everything CPU and serialize then ...

#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    //c10::Device device(c10::DeviceType::CPU);
    module = torch::jit::load(argv[1]); //, device);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

//module.to(at::kCPU);

  // Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::rand({1,4, 60, 60, 50})); // .to(at::kMPS));

// Execute the model and turn its output into a tensor.
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

}

/*

// https://pytorch.org/tutorials/advanced/cpp_export.html
// issues with moving FNO model to CPU in C++, but also in Python ... Follow up!!!!
// Detour: Save GPU/MPS trainied FNO via only weights state_dict. Read in with everything CPU and serialize then ...

#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    //c10::Device device(c10::DeviceType::CPU);
    module = torch::jit::load(argv[1]); //, device);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

//module.to(at::kCPU);

  // Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({1, 1, 60, 60})); // .to(at::kMPS));
*/

// Execute the model and turn its output into a tensor.
//at::Tensor output = module.forward(inputs).toTensor();
//std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
//}
