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

## 2026-11-14: makemore Part 1 (bigram and trigram)
**What I did:** Watched Karpathy's makemore Part 1 (1h57) and built
`02_makemore_bigram/02_makemore_bigram.ipynb`: counted bigrams in a
27×27 tensor, turned the counts into probabilities (+1 smoothing),
sampled names with `torch.multinomial`, and measured the average
negative log likelihood: 2.454 over 228,146 bigrams (a model that knows
nothing gives 3.30). Rebuilt the same model as a one-layer neural net
(one-hot input, weight matrix W, softmax) with gradient descent at
learning rate 50 and a penalty 0.01·mean(W²): loss 2.49 after 100 steps.
Exercise E01: a trigram neural net with 729 one-hot inputs (the two
previous letters) and 27 outputs, trained on the Kaggle GPU because the
CPU was too slow. After 100 steps its loss was 2.52, worse than the
bigram, so I trained for 1000 steps: loss 2.255 (last value printed, at
step 900), clearly below the bigram's 2.49 (neural net) and 2.454
(counting). All the neural-net losses include the small penalty term.
The sampled names look much more like names than after 100 steps.

**What broke / what confused me:**
- `torch.Generator` and `torch.multinomial`: what the seed does, and
  how a weighted die picks the next letter.
- The term `0.01*(W**2).mean()` in the loss: it pulls the weights
  toward 0, which does the same job as the +1 smoothing in the
  counting model.
- NLL: why the log and the minus sign.
- The trigram loss after 100 steps (2.52) was higher than the bigram's
  (2.49). Not a bug: the trigram has 729 rows instead of 27, so each row
  gets far fewer examples (228,146 / 729 is about 313 on average, versus
  about 8,450 for the bigram) and learns slowly. It needed more steps.

**What I learned:** A bigram model is a table of "which letter comes
next". The counting model fills the table from counts, and the neural
net learns the same table: each row of W is log-counts, and softmax
turns them back into probabilities. NLL is the average surprise per
letter: exp(−2.454) = 0.086, so the model behaves as if it were
guessing among about 11.6 equally likely letters (a blind guess among
27 gives 3.30). Smoothing, or the penalty on W in the neural net, keeps
unseen pairs from getting probability 0, which would make the loss
infinite. A trigram looks at two letters instead of one, so it can  
predict better, but its table is 27 times bigger and it trains more
slowly. When a loss looks worse than expected, check whether the model
has simply had too few steps before suspecting a bug.

**Open questions:** Does the trigram still win on names it was not
trained on? I have only measured it on the training names. And the table
grows 27 times with every extra letter of context, so how do real
language models look at much longer contexts?