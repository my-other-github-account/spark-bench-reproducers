# W3 e43-LUT cubin source

`gen_moe_w3.py` deterministically regenerates the six checked-in reference SASS files and `build.sh` assembles them with the open-source [cubit](https://github.com/kacper-daftcode/cubit) SM120 assembler pinned to commit `c139df8b34f1dcab607f8ccb685fdea948f3ae4d`.

```bash
./build.sh ./out
sha256sum ./out/*.cubin ../w3/*.cubin
```

The two hash sets must match by runtime filename.

The validated LUT is the e4m3-rounded asymmetric eight-level pair `MOEW3_LUT_LO=0xb6bfc6cd`, `MOEW3_LUT_HI=0x4d463c21`. Generation of all six SASS files was byte-compared against the campaign source before publication. The runtime cubins in `../w3/` match the generated e43 build byte-for-byte.
