#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>


torch::Tensor run_syncfree_strided_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);

torch::Tensor run_syncfree_fixed_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);


PYBIND11_MODULE(syncfree_attention, m)
{
    m.doc() = "My_fused_attention: My first attempt to fused attention (without any optimzation). ";
    m.def("run_syncfree_strided_attention", &run_syncfree_strided_attention, "fused sparse attention, Masked with Strided ");
    m.def("run_syncfree_fixed_attention", &run_syncfree_fixed_attention, "fused sparse attention, Masked with Fixed ");
} 