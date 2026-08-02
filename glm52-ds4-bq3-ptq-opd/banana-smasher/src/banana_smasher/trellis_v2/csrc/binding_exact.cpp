#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> trellis_v2_exact_cuda(
    const torch::Tensor& x,
    const torch::Tensor& lut_aos,
    const c10::optional<torch::Tensor>& overlap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "viterbi", &trellis_v2_exact_cuda,
      "QTIP L16/K2/V2 CUDA-Graph-replayed paired-step batch-16 LUT-reuse packed-backpointer exact Viterbi",
      py::arg("x"), py::arg("lut_aos"),
      py::arg("overlap") = c10::nullopt);
}
