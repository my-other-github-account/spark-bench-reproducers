"""Shipped-plane-bytes readers for the DS4 KLD candidate rail.

Inverse of the vllm-moet wire format packers:
  W2: moe_w2_planes.pack_fragment_major (2-bit codes, LUT {-4,-1,1,4},
      scales = ckpt block-32 UE8M0 bytes verbatim)
  W3: moe_w3_planes.pack_w3_plane (3-bit codes, LUT
      {-6,-3,-1.5,-0.5,.5,1.5,3,6}, scales re-derived amax->6.0 UE8M0)
Both use pack_scales [N//16,16,KS].transpose(1,2) layout.

unpack_* are roundtrip-asserted at import time against local copies of the
pack functions (transcribed verbatim from the shipped modules).
"""
import torch

W2_LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0])
W3_LEVELS = torch.tensor([-6.0, -3.0, -1.5, -0.5, 0.5, 1.5, 3.0, 6.0])


# ------------------------------------------------------------------ W2
def pack_fragment_major(codes):
    """verbatim moe_w2_planes.pack_fragment_major (for the self-test)."""
    N, K = codes.shape
    assert N % 16 == 0 and K % 64 == 0
    c = codes.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    c = c.view(-1, 4).to(torch.int32)
    packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6))
    return packed.to(torch.uint8).flatten()


def unpack_fragment_major(plane, N, K):
    """[N*K/4] u8 -> [N, K] u8 codes 0..3 (inverse of pack_fragment_major)."""
    nb, kb = N // 16, K // 64
    dev = plane.device
    b = plane.view(nb, kb, 8, 4, 2, 2, 2).to(torch.int64)
    codes = torch.stack([(b >> (2 * i)) & 3 for i in range(4)], dim=-1)
    # [nb,kb,g,t,tile,k32,half,k4] -> [nb,tile,g,kb,k32,half,t,k4]
    codes = codes.permute(0, 4, 2, 1, 5, 6, 3, 7).contiguous()
    return codes.view(N, K).to(torch.uint8)


# ------------------------------------------------------------------ W3
def pack_w3_plane(codes):
    """verbatim moe_w3_planes.pack_w3_plane (for the self-test)."""
    N, K = codes.shape
    assert N % 16 == 0 and K % 64 == 0
    dev = codes.device
    c = codes.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    c = c.view(N // 16, K // 64, 32, 8, 4).to(torch.int64)
    nb, kb = c.shape[0], c.shape[1]
    lo = c & 3
    low = (lo[..., 0] | (lo[..., 1] << 2) | (lo[..., 2] << 4)
           | (lo[..., 3] << 6)).to(torch.uint8)
    low = low.reshape(nb, kb, 256)
    hi = (c >> 2) & 1
    shift = (torch.arange(8, device=dev).view(1, 1, 1, 8, 1) * 4
             + torch.arange(4, device=dev).view(1, 1, 1, 1, 4))
    hw = (hi << shift).sum(dim=(3, 4))
    hib = torch.stack([(hw >> (8 * i)) & 0xFF for i in range(4)],
                      dim=-1).to(torch.uint8)
    hib = hib.reshape(nb, kb, 128)
    return torch.cat([low, hib], dim=2).reshape(-1)


def unpack_w3_plane(plane, N, K):
    """verbatim moe_w3_planes.unpack_w3_plane (device-safe)."""
    nb, kb = N // 16, K // 64
    dev = plane.device
    p = plane.view(nb, kb, 384).to(torch.int64)
    low = p[..., :256].reshape(nb, kb, 32, 8)
    lo = torch.stack([(low >> (2 * i)) & 3 for i in range(4)], dim=-1)
    hib = p[..., 256:].reshape(nb, kb, 32, 4)
    hw = (hib[..., 0] | (hib[..., 1] << 8) | (hib[..., 2] << 16)
          | (hib[..., 3] << 24))
    shift = (torch.arange(8, device=dev).view(1, 1, 1, 8, 1) * 4
             + torch.arange(4, device=dev).view(1, 1, 1, 1, 4))
    hi = (hw.view(nb, kb, 32, 1, 1) >> shift) & 1
    codes = (lo | (hi << 2)).view(nb, kb, 8, 4, 2, 2, 2, 4)
    codes = codes.permute(0, 4, 2, 1, 5, 6, 3, 7).contiguous()
    return codes.view(N, K).to(torch.uint8)


# ------------------------------------------------------------- scales
def unpack_scales(flat, N, KS):
    """inverse of moe_w2/w3_planes.pack_scales: [N*KS] u8 -> [N, KS] u8."""
    return (flat.view(N // 16, KS, 16).permute(0, 2, 1)
            .contiguous().view(N, KS))


def pack_scales(scales):
    N, KS = scales.shape
    assert N % 16 == 0
    return scales.view(N // 16, 16, KS).transpose(1, 2).contiguous().flatten()


def plane_dequant(codes, scale_bytes, levels):
    """codes [N,K] u8 + UE8M0 [N,K/32] u8 -> f32 [N,K]."""
    vals = levels.to(torch.float32).to(codes.device)[codes.long()]
    s = torch.exp2(scale_bytes.to(codes.device).float()
                   - 127.0).repeat_interleave(32, dim=-1)
    return vals * s


# --------------------------------------------------- import self-tests
torch.manual_seed(0)
_c2 = torch.randint(0, 4, (64, 128), dtype=torch.uint8)
assert torch.equal(unpack_fragment_major(pack_fragment_major(_c2), 64, 128), _c2)
_c3 = torch.randint(0, 8, (64, 128), dtype=torch.uint8)
assert torch.equal(unpack_w3_plane(pack_w3_plane(_c3), 64, 128), _c3)
_s = torch.randint(0, 255, (64, 4), dtype=torch.uint8)
assert torch.equal(unpack_scales(pack_scales(_s), 64, 4), _s)
del _c2, _c3, _s
