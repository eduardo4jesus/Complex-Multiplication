#include <torch/torch.h>

/********************************
 * Declaring Functions' Headings
 ********************************/
at::Tensor complex_multiplication_cuda_v1(at::Tensor, at::Tensor);

/**
 * Multiplies two tensors of Complex Tensors
 * @param x
 * @param h
 * @return ouput
 */
at::Tensor complex_multiplication(at::Tensor x, at::Tensor h) {
    printf("CPP: my_pytorch_extensions_cuda called\n");
    return complex_multiplication_cuda_v1(x, h);
}

/********************************
 * Binding Functions
 ********************************/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("complex_multiplication", &complex_multiplication, "CUDA kernel for multiplication of complex tensors");
}
