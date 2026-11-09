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

**Open questions:** My notes only do the chain rule for the last layer
with one neuron per layer. How does it extend to several neurons per
layer? And how does the gradient for the bias work?
