# Problem 2 - Make `QLoRA` work with `FSDP2`

> **NOTE:** I was unable to fully satisfy requirement #8 "You must enable all features in FSDP2 - ie showcase offloading, checkpointing, mixed precision training etc." I have enabled checkpointing and mixed precision training, but my submission does not have CPU offload enabled.

In addition to patching torch and bitsandbytes, I had to make some modifications to `accelerate`, which I directly pull in my notebook before running. Specifically I am pulling in my working branch in https://github.com/byi8220/accelerate/tree/experimental/qlora-fsdp2

### Expected Scoring Criteria Met

- ✅ if FSDP2_works_with_QLoRA:
- ❌   if torch_compile_works: B_score += 5
- ✅	  else: B_score += 3
- ❌	  if uses_part_A_and_single_kernel_and_faster: B_score += 3
- ➖   elif uses_torchAO:
- ➖     if torchAO_slower_than_BnB: B_score -= 3
- ➖ elif TP_or_PP_with_QLoRA:
- ➖   if zero_bubble: B_score += 3
- ➖   else: B_score += 2
- ➖ elif FSDP1_works_with_QLoRA: B_score += 1
- ✅ if kaggle_notebook_2_tesla_t4_example: B_score += 2

**Estimated Total Points: 5**

Getting FSDP2 working with QLoRA and accelerate is an incredibly involved process with many parts. I have provided a semi-funcitonal MVP.

### Summary

TODO: Add a writeup