# Lab log

Rules for writing entries: see [LAB_LOG_GUIDE.md](LAB_LOG_GUIDE.md).

## 2026-11-05: Visual intuition for neural networks
**What I did:** Watched 3Blue1Brown ch. 1–4 and took handwritten notes
(neurons and activations, weights and biases, sigmoid vs ReLU, cost
function, gradient descent, backprop with the chain rule). Worked a
one-weight example by hand (x=2, target 10) to see the loss, gradient
and learning rate. Full notes are in my paper notebook.

**What confused me:**
- Why we step against the gradient: the gradient points uphill, so the
  update subtracts it.

**What I learned:** A network is layers of numbers (activations). Each
neuron takes a weighted sum of the previous layer, adds a bias and
squashes it (sigmoid, or the simpler ReLU). Weights pick the pattern a
neuron looks for, and the bias sets how high the sum must be before it
fires. Learning means finding good weights and biases (13,002 in the
784-16-16-10 net). The cost measures how wrong the output is, and its
gradient says which way to push each weight and which ones matter most.
Backprop computes that gradient with the chain rule, from the output
backward. Training uses mini-batches because averaging over all the
data for every step is too slow.

**Open questions:** None.

## 2026-11-09: micrograd
**What I did:** Watched Karpathy's micrograd (2h25) and coded along in
`01_micrograd/01_micrograd.ipynb`: derivatives by nudging an input, a
`Value` class with backward for + * ** exp tanh, topological sort, then
Neuron, Layer, MLP and a training loop on 4 examples (loss 7.28 → 0.024
in 20 steps). Checks: `a + a` gives grad 2, the paper example gives
(6, −4, −2, 4), and my gradients match PyTorch. Then did the exercise
notebook (numerical gradients, softmax with negative log likelihood).
 
**What broke / what confused me:**
- What a derivative actually does: how much the output moves when I
  nudge one input by a tiny bit.
- How `backward` works: how each node passes its gradient back to its
  inputs, and why it runs in reverse order.
- Copied `draw_dot` from the video without studying it (display only).

**What I learned:** Backprop is the chain rule applied node by node:
each operation only knows its local derivative and multiplies it by
the gradient arriving from after it. Gradients must accumulate (`+=`)
because one value can feed several operations, and must be set to zero
before every backward pass in training. A gradient check against a
numerical estimate or PyTorch is how I know my code is right.
 
**Open questions:** In the softmax exercise the gradients came out as
the probabilities, except −0.886 for the correct class. Why is that
exactly "probability minus 1"? And how does PyTorch do this on whole
tensors instead of single numbers?