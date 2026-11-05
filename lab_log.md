<!--
LAB LOG RULES (read before writing an entry)

WHEN: one entry per study session or per tracker step, written the same day
while it is fresh. Add new entries at the BOTTOM. Never rewrite old entries
(only add a "Closed:" line under an open question when you answer it).

SIZE: 6-12 lines. If it is longer, you are copying notes instead of
summarising. Detailed notes, formulas and drawings stay on paper or in code.

VOICE: your own words, short sentences, no text copied from videos or AI.
Write so that you in 3 months, without the video, can understand it.

HEADING: "## YYYY-MM-DD: short topic"

THE FOUR PARTS

**What I did**
  Facts only: what I watched, read, built or ran, and the results with
  numbers (accuracy, loss, time). Say where the full notes or code live
  ("paper notebook", file name). No opinions here.

**What broke / what I looked up** (or "What confused me")
  Every error, bug, wrong guess and thing I had to look up or ask for help
  with. Write the cause or the fix in one line each. This is the most
  valuable part: be honest, include the stupid mistakes. If nothing broke,
  write what felt hard.

**What I learned**
  The ideas I could now explain to a friend, 2-5 sentences, no formulas.
  Say what changed in my understanding, not just a topic name.
  Test: if I could have written it before the session, it is not a lesson.

**Open questions**
  Things I still don't get or want to check. Be specific ("how does the
  chain rule work with several neurons per layer?"), not "backprop".
  Write "none yet" if truly none. Later, answer them and add "Closed:".

TEMPLATE
  ## YYYY-MM-DD: topic
  **What I did:** ...
  **What broke / what I looked up:**
  - ...
  **What I learned:** ...
  **Open questions:** ...
-->

# Lab log

## 2026-11-02: PyTorch basics
**What I did:** Watched the AssemblyAI crash course. Wrote a linear
regression by hand, then an MNIST classifier, then FashionMNIST from
a blank file. Test accuracy: 86.6%.

**What broke / what I looked up:**
- Dataset parameters (root, train, transform, download)
- forward(self, x) needs x as a parameter
- criterion is built first, then called; torch.optim is not part of nn
- len(test_loader) counts batches, not images (gave 8659%)
- criterion needs labels as a tensor, not a plain int

**What I learned:** backward() fills .grad, step() uses it,
zero_grad() clears it. CrossEntropyLoss includes softmax, so the
model outputs raw scores.

**Open questions:** none yet

## 2026-11-05: Visual intuition for neural networks
**What I did:** Watched 3Blue1Brown ch. 1–4 and took handwritten notes
(neurons and activations, weights and biases, sigmoid vs ReLU, cost
function, gradient descent, backprop with the chain rule). Worked a
one-weight example by hand (x=2, target 10) to see the loss, gradient
and learning rate. Full notes are in my paper notebook.

**What confused me:**
- Why we step against the gradient: the gradient points uphill, so the
  update subtracts it.
- Learning rate: too big overshoots (w went 1 → 17 instead of settling
  at 5), too small is just slow.

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
