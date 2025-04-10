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

### Comments

This problem was split into multiple notebooks, in order to better isolate memory measurement. Additionally, for the largest test case in `MemEffLinear.ipynb` where `bsz = 4, qlen = 4096, hd = 4096, vocab = 128K`, I used an A100 to demonstrate a memory usage reduction from 25GB -> 11.30 GB. This was isolated since A100 compute is expensive, and I didn't want to burn that on the other problems.

`MemEffLinear.ipynb` is the landing notebook, demonstrating 50% memory reductions with a MemEffLinear implementation.

`Unsloth_Problem_5_Other_Loss_Functions.ipynb` shows other loss functions besides cross entropy loss.

`Unsloth_Problem_5_Llama_1B.ipynb` compares the training loss between MemEffLinear and regular linear.

`Unsloth_Problem_5_Other_Loss_Functions.ipynb` completes 250 steps of a GRPO training run on Llama 3.1 8B, just to demonstrate it works. This required some slight modification to the signature to get the `MemEffLinear` module to accept `selective_log_softmax`.