## Problem 5 - Memory Efficient Backprop

### Expected Scoring Criteria Met

- ✅ if VRAM_50_percent_reduction: E_score += 2
- ✅ if remove_float32_upcast: E_score = 0
- ✅ if show_ce_loss_works: E_score += 1
- ✅ if show_other_functions_work: E_score += 1
- ✅ if hardcoded_gradients: E_score = 0
- ✅ if allows_dynamic_chunk_sizes: E_score += 1
- ✅ if llama_1B_training_loss_matches: E_score += 1
- ✅ if GRPO_memory_efficient_linear_works: E_score += 4

**Estimated Total Points: 10**

`llama_1B_training_loss_matches` and `GRPO_memory_efficient_linear_works` have their own separate colab notebooks.

### Summary

TODO: Add a writeup