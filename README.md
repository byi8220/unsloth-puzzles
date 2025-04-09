# unsloth-puzzles

My attempt at solutions to [Unsloth puzzles](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH).

## What is this repo?

[Unsloth](https://github.com/unslothai/unsloth) is an ML library built on top of standard ML frameworks such as Pytorch and Huggingface, which aims to patch existing code for the sake of performance optimization.

The Unsloth org has published a [problem set](https://x.com/danielhanchen/status/1891194528931209644) which highlights uses cases with room for performance improvement.

- Problem 1 - Convert `nf4` to Triton
- Problem 2 - Make `QLoRA` work with `FSDP2`
- Problem 3 - Make `torch.compile` work without graph breaks for QLoRA
- Problem 5 - Memory Efficient Backprop

As someone pretty new to ML development, these problems were quite challenging and open ended. It was uncertain how productionized the solutions should be, and the code in these notebooks are messy prototypes.

