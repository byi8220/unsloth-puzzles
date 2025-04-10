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

`tested_in_f16_and_bf16`: Requirements as written, this kernel *cannot* operate on `bf16` on an Nvidia T4. We would have to use a newer GPU if we want to verify this. One would have to run this notebook on an L4 colab instance, although we lose the performance edge when we move there, as the reference dequantize functions perform better on L4 than T4.

These are just my judgements. I could be wrong, as sometimes the requirements are a bit vague.

### Comments

The first problem is to implement an NF4 dequantization kernel in Triton. NF4, or `NormalFloat4`, is a quantized data type that compresses floating point values and represents them as 4-bit values. The memory savings from this are immense, as compressing 16-bit or 32-bit values into 4 bits is a 4x or 8x reduction, respectively.

There are already many [great resources](https://www.youtube.com/watch?v=2ETNONas068) on this topic, so skipping over the details, a quantized tensor's elements are split into blocks of size `B`, and those blocks are further split into double blocks of size `B2`, and the formula for dequantizing the `i`th element of a quantized tensor is:

```
dequantized_values[i] = code[quantized_values[i]] * (code2[absmax[i//B]] * absmax2[i//(B*B2)] + offset)
```

Importantly, note that computing the `i`th element can be done without any dependencies on any other elements, making this problem embarassingly parallel. However, multiple elements share the same `absmax` and `absmax2` values, so there is opportunity to save on loads if we size our grid properly. 

For this problem, I chose to make each program instance process 64 separate blocks, which leads to a 15% speedup. 

I also experimented with a parameter sweep, showing that a poor selection of parameters could lead to slowdown.

The requirements for this problem were straightforward, but tough. One particularly tricky requirement is the conflict between needing to run on a T4, and needing it to work on bfloat16. Nvidia T4 GPUs do not support T4 calculation.

During some runs, my kernel is outperforming the reference implementation by over 30%! On average, with carefully selected launch params, it hovers around 1.15x speedup. This is a noticable improvement, which begs the question: what's the catch? 

For one, we do significantly loosen the tolerance. Even more strangely, compiling the kernel caused a massive slowdown.