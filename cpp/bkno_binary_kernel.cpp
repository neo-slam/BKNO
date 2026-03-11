#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

inline int64_t popcnt64(uint64_t x) {
#if defined(_MSC_VER)
  return static_cast<int64_t>(__popcnt64(x));
#else
  return static_cast<int64_t>(__builtin_popcountll(x));
#endif
}

inline int64_t binary_dot_xnor_popcnt(
    const uint8_t* a,
    const uint8_t* b,
    int64_t len) {
  int64_t pop_a = 0;
  int64_t pop_b = 0;
  int64_t pop_xnor = 0;

  for (int64_t base = 0; base < len; base += 64) {
    const int64_t bits = std::min<int64_t>(64, len - base);
    uint64_t a64 = 0;
    uint64_t b64 = 0;

    for (int64_t i = 0; i < bits; ++i) {
      const uint64_t abit = static_cast<uint64_t>(a[base + i] != 0);
      const uint64_t bbit = static_cast<uint64_t>(b[base + i] != 0);
      a64 |= (abit << i);
      b64 |= (bbit << i);
    }

    const uint64_t mask = (bits == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1);
    const uint64_t xnor = ~(a64 ^ b64) & mask;

    pop_a += popcnt64(a64);
    pop_b += popcnt64(b64);
    pop_xnor += popcnt64(xnor);
  }

  // From paper:
  // <a,b> = (POPCNT(a) + POPCNT(b) + POPCNT(XNOR(a,b)) - n) / 2
  return (pop_a + pop_b + pop_xnor - len) / 2;
}

}  // namespace

torch::Tensor binary_conv3d_forward(
    torch::Tensor input_bits,   // [N, C, D, H, W], uint8/bool, values in {0,1}
    torch::Tensor weight_bits,  // [O, C, kD, kH, kW], uint8/bool, values in {0,1}
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w) {
  TORCH_CHECK(input_bits.device().is_cpu(), "input_bits must be on CPU.");
  TORCH_CHECK(weight_bits.device().is_cpu(), "weight_bits must be on CPU.");
  TORCH_CHECK(input_bits.dim() == 5, "input_bits must be 5D [N,C,D,H,W].");
  TORCH_CHECK(weight_bits.dim() == 5, "weight_bits must be 5D [O,C,kD,kH,kW].");

  auto in = input_bits.to(torch::kUInt8).contiguous();
  auto wt = weight_bits.to(torch::kUInt8).contiguous();

  const auto N = in.size(0);
  const auto C = in.size(1);
  const auto D = in.size(2);
  const auto H = in.size(3);
  const auto W = in.size(4);

  const auto O = wt.size(0);
  const auto Cw = wt.size(1);
  const auto kD = wt.size(2);
  const auto kH = wt.size(3);
  const auto kW = wt.size(4);

  TORCH_CHECK(C == Cw, "Channel mismatch: input C != weight C.");

  const int64_t outD = D;
  const int64_t outH = H;
  const int64_t outW = W;

  auto out = torch::zeros({N, O, outD, outH, outW}, torch::TensorOptions().dtype(torch::kFloat32));

  const auto in_acc = in.accessor<uint8_t, 5>();
  const auto wt_acc = wt.accessor<uint8_t, 5>();
  auto out_acc = out.accessor<float, 5>();

  const int64_t kSize = C * kD * kH * kW;
  std::vector<uint8_t> patch;
  std::vector<uint8_t> kern;
  patch.resize(static_cast<size_t>(kSize));
  kern.resize(static_cast<size_t>(kSize));

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t o = 0; o < O; ++o) {
      // Flatten one kernel for reuse.
      int64_t idx = 0;
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t kd = 0; kd < kD; ++kd) {
          for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
              kern[static_cast<size_t>(idx++)] = wt_acc[o][c][kd][kh][kw];
            }
          }
        }
      }

      for (int64_t od = 0; od < outD; ++od) {
        for (int64_t oh = 0; oh < outH; ++oh) {
          for (int64_t ow = 0; ow < outW; ++ow) {
            int64_t pidx = 0;
            for (int64_t c = 0; c < C; ++c) {
              for (int64_t kd = 0; kd < kD; ++kd) {
                for (int64_t kh = 0; kh < kH; ++kh) {
                  for (int64_t kw = 0; kw < kW; ++kw) {
                    const int64_t id = od + kd - pad_d;
                    const int64_t ih = oh + kh - pad_h;
                    const int64_t iw = ow + kw - pad_w;
                    uint8_t v = 0;
                    if (id >= 0 && id < D && ih >= 0 && ih < H && iw >= 0 && iw < W) {
                      v = in_acc[n][c][id][ih][iw];
                    }
                    patch[static_cast<size_t>(pidx++)] = v;
                  }
                }
              }
            }
            const int64_t dot = binary_dot_xnor_popcnt(
                patch.data(), kern.data(), kSize);
            out_acc[n][o][od][oh][ow] = static_cast<float>(dot);
          }
        }
      }
    }
  }

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "binary_conv3d_forward",
      &binary_conv3d_forward,
      "Binary conv3d forward (XNOR/POPCNT-style).");
}

