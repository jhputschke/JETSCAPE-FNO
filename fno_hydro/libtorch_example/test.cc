#include <torch/torch.h>
#include <iostream>

int main() {

//torch::Tensor tensor = torch::rand({2, 3});
//std::cout << tensor << std::endl;

// Create a 2x3 tensor
auto tensor = torch::tensor({{1, 2, 3}, {4, 5, 6}});
std::cout << "Original tensor:\n" << tensor << std::endl;

// Repeat the tensor twice along dimension 0 and once along dimension 1
auto repeated_tensor = tensor.repeat({2, 1});
std::cout << "Repeated tensor (2x1):\n" << repeated_tensor << std::endl;

// Repeat the tensor twice along dimension 1 and once along dimension 0
auto repeated_tensor_2 = tensor.repeat({1, 2});
std::cout << "Repeated tensor (1x2):\n" << repeated_tensor_2 << std::endl;

// Repeat the tensor twice along dimension 0 and three times along dimension 1
auto repeated_tensor_3 = tensor.repeat({2, 3});
    std::cout << "Repeated tensor (2x3):\n" << repeated_tensor_3 << std::endl;

return 0;
}
