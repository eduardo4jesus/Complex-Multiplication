# Complex Multiplication

In this Repository you will a Pytorch Extension for an element-wise complex multiplication that considers two tensors, `x` and `h`.

## Operation

1. Given

- `x`, with shape `(B, C, H, W)`; and
- `h`, with shape `(B, F, H, W)`


2. This multiplication operation consists in, first, expanding 
- `x` to `(B, 1, C, H, W)`; and
- `h` to `(B, F, 1, H, W)`.

3. Then, the element-wise complex multiplication is done, and the dimension 2 is accumulated (`.sum(2)`). Thus, resulting in
- `output`, with shape `(B, F, H, W)`.

# Files

- `/pytorch_extension`: Contains all files for the Pytorch extension (`.cu`, `.cpp.` and `setup.py`).
- `notebooks/Demonstration.ipynb`: Demonstration on how to import and use the Pytorch extension.
- `Makefile`: simply call the `make` command in the terminal to build the extension.

## NOTE

By no means this is meant to be a complete extensions used for production. I removed all checks to focus only on the code.
This is only a code for demonstration purposes only. The implementation is intentionally very simple and non-optimum.

This code was inspired in the ones from [adam-dziedzic](https://github.com/adam-dziedzic/bandlimited-cnns/tree/master/cnns/nnlib/pytorch_cuda/complex_mul_cuda).
However, in our code, we addressed the compilation warnings, which required [using Accessors](https://pytorch.org/tutorials/advanced/cpp_extension.html#using-accessors).
