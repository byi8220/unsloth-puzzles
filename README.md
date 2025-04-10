# unsloth-puzzles

My attempt at solutions to [Unsloth puzzles](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH).

## What is this repo?

[Unsloth](https://github.com/unslothai/unsloth) is an ML library built on top of standard ML frameworks such as Pytorch and Huggingface, which aims to patch existing code for the sake of performance optimization.

The Unsloth org has published a [problem set](https://x.com/danielhanchen/status/1891194528931209644) which highlights uses cases with room for performance improvement.

As someone pretty new to ML development, these problems were quite challenging and open ended. It was uncertain how productionized the solutions should be, and the code in these notebooks are messy prototypes.

### Direct Backup Links To Notebooks

In case the previews in github aren't rendering properly, direct links are included below.

I added this section as as github literally broke this feature recently: https://github.com/orgs/community/discussions/155944

Problem 1 - Convert `nf4` to Triton
- https://colab.research.google.com/drive/1q4rVHD8yY95lievqXDp3PnLfLHy6pDHX

Problem 2 - Make `QLoRA` work with `FSDP2`
- https://www.kaggle.com/code/byi8220/unsloth-problem-2-kaggle-fsdp2-fixed-for-gh

Problem 3 - Make `torch.compile` work without graph breaks for QLoRA
- https://colab.research.google.com/drive/1x3uyg0kSBpBHs3dZnshXdUje32ItLLXU

Problem 5 - Memory Efficient Backprop
- https://colab.research.google.com/drive/1mflC5nxwqboWBn7MwlQZV9VI02X433iB
- https://colab.research.google.com/drive/1vEzZL-xGL6k1Y-NuRdrlNudzMFA2fQ0R
- https://colab.research.google.com/drive/1yBflfWmBGJUBtFAk7tmzX350juqAguc8
- https://colab.research.google.com/drive/1MmN1DAj3VXxd_EjwSE7cQsqL-93lHbzi