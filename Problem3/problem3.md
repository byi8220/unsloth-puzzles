# Problem 3 - Make `torch.compile` work without graph breaks for QLoRA

### Expected Scoring Criteria Met

- ✅ if uses_flex_attention:
- ✅   if dynamic_sequence_length_works: C_score += 3
- ➖   else: C_score += 1
- ❓ if no_torch_compile_BnB: C_score -= 2
- ➖ elif use_part_A: C_score += 1
- ❓ elif torch_compile_BnB: C_score += 1
- ✅ if attention_compiled:
- ➖ if excessive_recompilation: C_score -= 3
- ✅else: C_score += 2
- ✅ if mlp_compiled:
- ➖ if excessive_recompilation: C_score -= 3
- ✅   C_score += 1
- ➖ if not loss_compiled: C_score -= 1
- ➖ if not layernorms_compiled: C_score -= 3
- ✅ if max_autotune_triton_matmul:
- ➖   if excessive_recompilation: C_score -= 2
- ✅   else: C_score += 2

**Estimated Total Points: 6-9**

I'm not fully sure what `no_torch_compile_BnB` and `torch_compile_BnB` are asking for (or why they are separate criteria). I compiled what I could.

### Comments

A huge thanks to [RameshBabuAsh](https://github.com/RameshBabuAsh) and [Ghogha_Atif](https://discuss.pytorch.org/u/Ghogha_Atif/), who were also working on the Unsloth Puzzles. We seem to have all discovered the same [issue](https://discuss.pytorch.org/t/how-to-solve-the-graph-break-happen-in-torch-compile/216858/) with `Params4bit`, which was a bug that was fixed in torch nightly.
