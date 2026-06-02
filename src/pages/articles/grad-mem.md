---
layout: ../../layouts/MarkdownLayout.astro
title: "Part 1. Building the Baseline"
date: "2026-05-23"
description: "Exploring Meta-Learning, identifying shortcomings, & getting through training instability"
tldr: "I build a faithful reproduction of the original paper and try to further extrapolate to other tasks, tackling issues of training instability and faulty implementation details along the way. In the process, I created a custom Hessian-Vector Product (HVP) kernel to make the double backwards pass tractable on my machine by reducing memory overhead by ~50x over vanilla PyTorch graph materialization. Training results and further evaluation come in the next article."
project: "gradmem"
tags: ["gradmem"]
---
## Methods Used
## Background
For this project, I implemented and extended the methodology presented in this [recent paper](https://arxiv.org/pdf/2603.13875v1), which outlines an extension to the [seminal paper](https://arxiv.org/pdf/1703.03400) on Meta-Learning. Typical Meta-Learning in an NLP setting poses a uniform loss for both the inner and outer loops, wherein only the weights of the network are trainable parameters. GradMem departs from this baseline in an exciting way that motivates a new area of study.

Most importantly, it deviates from the original formulation in its objectives. The inner loop (coined as the WRITE phase in the paper) encourages *compression* of information into a new concept called "Memory Tokens". These tokens function akin to [gist tokens](https://arxiv.org/pdf/2304.08467) in that they are trainable, high dimensional vectors that are the focus of the WRITE phase's compression task. They are prepended as context to the user's query during the READ phase, and prepended to the actual context in the WRITE phase. Crucially, it is **only** these Memory tokens that are "learned" during the WRITE phase - the network's weights remain frozen. This is the only phase that is run at Test-Time, to update the memory tokens to absorb information related to the provided context. 

On the other hand, the outer loop's loss instead measures task performance (in the NLP setting, this would be faithfulness to the Ground Truth). My understanding is that the objective of this phase is to induce a strong prior into the Memory Tokens so that the model doesn't have to learn a representation of these abstract tokens from scratch - it is given a baseline that has *already* soaked in some information. Along with this, it also functions as a way to teach the network a new task - how to *route information* into these tokens during the inner loop. The outer loop has gradients that flow from the unrolled inner loops as well, which is the mechanism through which the weights become cognizant of the impact on it's routing strategies on the inner loop.

The READ phase follows the test-time training stage, which is just a normal forward pass using the memory tokens and standard autoregressive generation. The results of the paper focus primarily on an unusual benchmark, but perhaps fitting for this regime - synthetic KV retrieval. Given a randomized, nonsensical (in isolation) Key-Value pair of strings, the model must predict the Value of a specific Key (which is passed as the user's query). The context itself is removed, and the model must use just the memory tokens that were updated at test-time. An example would be `!OZ:Tr!!Ih:hA!!L3:xu!!tU:7d!|` as an input, where the "query" is `?!tU:`, and the target is `7d!|` (taken directly from their training set posted on HuggingFace). Along with this, the paper reported positive evals on Short SQuAD run using a 175M model on an input length of 40 tokens.

Now, my immediate thoughts after reading the paper were "I want to run this on a more rigorous NLP task". Achieving *accurate* compression through a Meta-Learned mechanism ? That falls exactly in line with the objectives I had written about on my home page ! To be clear, the paper *does* emphasize why they opted for this task rather than Natural Language tasks (it has to do with their design of memory tokens), but more on this later. First, a quick detour to another *massive* issue that this training regime surfaces.

## The Memory Cost of Meta-Learning
One thing that I had hinted at in a previous paragraph was the fact that gradients had to flow from the unrolled inner loops into the outer loop. This is a massive oversimplification of what is actually needed to make this feasible for my setup. This unrolling of loops requires the *entire* higher-order graph to be materialized - which is needed to backprop the derivatives from the WRITE loop into the outer loop's loss propagation. The inner loop loss and the resulting memory token updates are *contingent on $\theta$*, and the outer loop is in turn contingent on these newly updated memory tokens. When taking the derivative of the Loss with respect to the $\theta$, due to the chain rule, we need to further apply the derivative to the derivative of the inner loop's loss.

To avoid the complete materialization, I compute the Hessian-Vector Product(HVP) using a fused kernel that was built on top of what the original authors had already contributed [here](https://github.com/yurakuratov/gradmem/blob/main/attn_double_bwd/hvp_semi_manual.py). They wrote the differentiable attention primitives, but I completed the HVP integration that makes it usable for an actual double-backward pass. I want to be clear that this kernel was built entirely on top of their initial effort. I do not want to make any claims of novelty.

I extended it beyond their initial manual implementation *solely through the use of AI*. I had never worked at the kernel level before, and GradMem's algorithm required materializing double-backward (Hessian-Vector product) graphs, unrolled across multiple inner loop optimization steps. To bypass the enormous memory footprint of PyTorch's native `create_graph=True` pipeline, I used Claude to help finish a fused Reverse-over-Reverse (RoR) HVP Triton kernel that is far more compute friendly. Rather than relying on PyTorch's autograd to materialize the higher-order execution graph of the inner loop(s), the kernel bypasses this entirely by evaluating the HVP using a fused subgraph. This avoids retaining a per-step (denoted by the K value in the below tables) higher-order graph, so memory stays near flat as K grows.

Following this, I extended it further towards a Forward-over-Reverse (FoR) implementation; I found this necessary as a quick profiling run of the kernel showed that 49% of the total CUDA time was spent in `_attn_double_bwd_q_kernel`, and another 40% in `_attn_double_bwd_kv_kernel`. To mitigate this, and the graph materialization *entirely*, I pointed Claude to [this blog post](https://iclr-blogposts.github.io/2024/blog/bench-hvp/) containing the implementation details for the FoR kernels. This implementation yielded even more drastic improvement, but came with its own set of issues. I have set it aside for the time being to opt for progress in a direction where I have more control - actually implementing GradMem.

## Kernel Benchmarks
The figures shown below are obtained from running the tests on an **H200 SXM** with shapes `B=1, H=32, S=8192, head_dim=128, bf16`, on a full WRITE loop (the inner loop described in GradMem) comparing against PyTorch's eager `create_graph=True` double-backward pass. To be clear, the actual training runs I am testing with are on sequence lengths far smaller than 8192 (validating my approach and implementation) - so the memory savings are proportionally less, but will become a lot more noticeable once I begin to scale up.

### Reverse-over-Reverse vs. Eager PyTorch
| K (inner steps) | Kernel (ms) | Eager (ms) | Speedup | Kernel VRAM | Eager VRAM | Memory ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 158.0 | 70.1 | 0.44x | 836.00 MiB | 44.25 GiB | 0.018x |
| 2 | 314.4 | 137.8 | 0.44x | 965.00 MiB | 52.25 GiB | 0.018x |
| 3 | 466.6 | 207.0 | 0.44x | 1.07 GiB | 60.25 GiB | 0.018x |
| 4 | 621.8 | 274.4 | 0.44x | 1.19 GiB | 68.25 GiB | 0.017x |

Here, we can see a strong reduction of up to **~57x** as K increases. However, it is quite compute inefficient - displaying a **slowdown of 2.26x**.

### Forward-over-Reverse vs. Eager PyTorch
| K (inner steps) | Kernel (ms) | Eager (ms) | Speedup | Kernel VRAM | Eager VRAM | Memory ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 119.9 | 69.3  | 0.58× | 836 MiB | 44.25 GiB | 0.018× |
| 2 | 176.5 | 137.1 | 0.78× | 836 MiB | 52.25 GiB | 0.016× |
| 3 | 234.0 | 204.6 | 0.87× | 836 MiB | 60.25 GiB | 0.014× |
| 4 | 290.3 | 271.4 | 0.93× | 836 MiB | 68.25 GiB | 0.012× |

This paints an even rosier picture. We can see that for the FoR kernel, the peak memory stays **constant at 836 MiB across all K** because the graphs are *never* materialized, whereas the eager implementation grows linearly with the number of unrolled inner steps (K) - reaching 68.25 GiB at K=4. This is an **~80x memory reduction**. The kernel reaches performance parity with the eager implementation as K grows, evidenced by the 0.93x speedup (or 1.07x slowdown) at K=4. It trades a small amount of compute for the memory headroom that would otherwise OOM for larger K / sequence lengths.

It should be noted that the accuracy of this kernel was validated **only for the Qwen3 family of models**. I made model-centric changes that traded generalizability for efficiency for my use case - a tradeoff I was okay making in order to get the ball rolling. As a result of this, the kernel produced NaNs on architectures it was not tuned for, namely the Pythia model that the original paper used.

Having eased the memory burden and validated the accuracy of the kernel on the model I was using, I could finally begin implementing the paper !

## Troubleshooting & Methodology Divergence
As mentioned before, I encountered some very interesting phenomena wherein the memory tokens simply *refused* to retain information. I ran hyperparameter sweeps to figure out if it was an issue with my settings or implementation. Amongst all these experiments, a few things became even clearer to me. Training the outer loop was *incredibly* unstable at larger values of K - I had to aggressively incorporate clipping and normalization terms to not immediately degenerate into NaNs. I was also able to identify the ideal K/alpha (inner loop learning rate) values that didn't lead to diverging inner loop updates

Aside from that, a host of issues came with integrating the kernel itself - ranging from autocasting issues leading to the monkey-patched kernel insertion being bypassed entirely (the benchmarks were measured after this was resolved), to dtype mismatch when materializing the fused subgraph using `create_graph=True`, having the HuggingFace library bypass the monkey-patch entirely by directly calling `F.scaled_dot_product_attention` internally, among others.

In the end, I couldn't find a definitive culprit for the representational collapse witnessed in the memory tokens, but during my search, a question crossed my mind. In the task formalization, the authors provide both the memory tokens **and the context** during the inner WRITE phase. Why were we doing this ? How does it ensure the model is actually compressing information faithfully ? 

The paper states that the memory "encode[s] information about [the context] that is not predictable from the prefix $t_{<i}$ alone". This works well for random KV tasks, as the model's prior is not able to predict anything from the random prefix (which refers to the input context until timestep i) - so memory **has** to encode everything. 

For Natural Language tasks, this does not necessarily hold. The model is able to infer from the prefix alone, so almost *nothing* is learned by the memory tokens. My own tests with this confirmed as much - in fact, using random values for the memory tokens actually produced a lower loss than the "learned" tokens in 1 instance. If I wanted to make this work for my use case, I would need to make changes to how the task was designed. It also begged the question - does this CE loss loss actually achieve the goal of compression ? I explore more along this avenue in my next article.