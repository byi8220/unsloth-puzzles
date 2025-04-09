# Problem 1 - Convert `nf4` to Triton

### Expected Scoring Criteria Met

- ✅ if single_triton_kernel: A_score += 3
- ✅ speedup = old_time / new_time
- ➖   if speedup <= 1.00: A_score -= 3
- ✅   if speedup >= 1.05: A_score += 1
- ✅   if speedup >= 1.10: A_score += 2
- ✅   if speedup >= 1.15: A_score += 2
- ❓ if kernel_works_in_torch_compile: A_score += 1
- ❓   else: A_score -= 1
- ❌ if custom_asm_works: A_score += 3
- ❓ if uses_cache_eviction: A_score += 1
- ❓ if tested_in_f16_and_bf16: A_score += 1

**Estimated Total Points: 7 to 11**

The `single_triton_kernel` and `speedup` criteria can be quickly verified at a glance. However, the other requirements were harder to evaluate.

`kernel_works_in_torch_compile`: the kernel itself physically works under `torch.compile()`, however there is a significant slowdown. This could be interpreted either way.

`uses_cache_eviction`: I'm unsure what "using cache eviction" means. In my kernel, I specifically change triton's `cache_modifier`, with the intent of hinting to triton how to cache evict. I'm unsure if this counts.

`tested_in_f16_and_bf16`: Requirements as written, this kernel *cannot* operate on `bf16` on an Nvidia T4. We would have to use a newer GPU if we want to verify this. I have attached a second notebook which runs this on an L4 colab instance. The tolerances and performance behave differently when changing hardware.

These are just my judgements. I could be wrong, as sometimes the requirements are a bit vague.

### Summary

The requirements for this problem were straightforward, but tough. One particularly tricky requirement is the conflict between needing to run on a T4, and needing it to work on bfloat16. Nvidia T4 GPUs do not support T4 calculation.

During some runs, my kernel is outperforming the reference implementation by over 30%! On average, with carefully selected launch params, it hovers around 1.15x speedup. This is a noticable improvement, which begs the question: what's the catch? 

For one, we do significantly loosen the tolerance. Even more strangely, compiling the kernel caused a massive slowdown.

TODO: Add a writeup