# Runtime and container closure status

This is a receipt-backed point-in-time snapshot of the P970/P987/P991/P993/P994/P997 serving chain. It is not a live-service guarantee. Machine-readable pins are in `artifacts/RUNTIME_CONTAINER_CLOSURE.public.json`.

## P970 product status

The legacy P970 parent card remained blocked, but its successor chain reached a real 43-layer OpenAI-compatible endpoint in P994. The endpoint passed health, model listing, and completion requests with:

- exact grouped-loader source commit `401ed71ec939a60dd813547b8905933b40dc8046`;
- exact pack manifest `3650fe7e627b180a979fb8304f90e888333671cf03334e965fd5b14b7393b220`;
- `109,921,775,107` resident bytes, within the 110-GiB gate;
- zero swap;
- all four routes nonzero;
- first-response receipt `33a9f5408032c0969519e46ce2cc3282d2940234ddd999e7f96da5acb9ddc4ae`;
- endpoint-winner receipt `4d5f9e75e5e94272719634c8528e558643eeb62ea2d3812cbb0b1ab20dadf343`;
- measured first response: `111.5344773982364` prefill tok/s, `27.269016424315677` decode tok/s, and `0.115520084000309` s TTFT.

The first response is an API/runtime proof, not a semantic-quality result. Canonical evaluation was still running at this snapshot.

## P987 loader integration

P987 is blocked for review, not because the grouped loader failed. The minimal loader commit and exact pack passed 32/32 full tests, 15/15 focused tests, and physical CUDA/HTTP gates for L000/L004/L042. The original image then failed closed before CUDA/HTTP bind because its vLLM 0.24 extension required torch 2.11 while the image supplied torch 2.10. The blocker receipt is `8a71e05d27dd59f0597ef5742a4494325ddb8055afd8408e8b61b0bab4b96322`. No endpoint, response, or TPS claim belongs to that original image.

## P991 compatible runtime

P991 completed an adoptable torch 2.11/CUDA 13 runtime closure and passed exact P987 L000/L004/L042 CUDA/HTTP gates. The closure receipt is `77265816f1453cc4767ead29e79a1270702ad7b0684d421ed5e667ee63d715e7`; its relocation archive is `c7f4fe70ce713f525012de083a163d05f91871069ec2cbe5c2e004ffe4df7114`.

That lane did **not** produce an endpoint: all 43 layers constructed, but the strict residency gate observed `89,551,425,418` of `102,867,186,131` required bytes and failed closed before HTTP bind.

## P993 ABI closure

P993 completed a transferable, version-aligned torch overlay. The ABI seal is `7fcacaeba2492b33af936abee3c595d80fa29e5084f662ccdea230d46cd607d9`; the overlay is `21107495d56719a98f03346b3e2cf777b8943e3ec66080f347ae98d4fe8c9c1b`. It imported the unchanged vLLM extension on CUDA and reproduced the canonical L004 grouped CUDA/HTTP output identity. It is an L004 ABI closure, not a full endpoint receipt.

## P997 golden-container status

The P997 parent was blocked while its build, gate-contract, update-process, and promotion successor lanes continued. Candidate receipt `cb80f64381d79aa959bece211c0984ddf65aefda9835ae55dcce7aa6ffc25a83` proves the stable-libtorch extension import and CUDA context inside image `sha256:4ef72d6a00ec97458dd77349785a0d0912cd0337683ee818b48b0921af2a9c5e`, but the candidate was explicitly reclassified as **G1 incomplete** because the separately mandated legacy `vllm._C` import is absent. It is not a G1 pass.

No G1-G7 terminal receipt set or golden tag exists. Independently, the first physically ready endpoint identity failed the configured G6 prefill floor: observed `111.5344773982364` tok/s versus the required `1000` tok/s. Therefore the correct publication status is **candidate only; G1 incomplete; G6 failed on the endpoint winner; no golden promotion**.

## Publication boundaries

- A layer-gate pass is not an endpoint pass.
- A first API response is not a semantic-quality result.
- A candidate image is not golden until all configured promotion gates pass inside that image.
- This snapshot does not publish any unsealed repair-improvement claim.
