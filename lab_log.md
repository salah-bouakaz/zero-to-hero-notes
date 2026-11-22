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

## 2026-11-19: makemore Part 2 (MLP)
**What I did:** Watched Karpathy's makemore Part 2 (1h15) and built
`03_makemore_mlp/03_makemore_mlp.ipynb`: built a character-level MLP from scratch in PyTorch (3-character context predicting the next character) with a learned embedding table `C`, a `tanh` hidden layer, and cross-entropy loss. Split the dataset 80/10/10 (train/dev/test), used an exponential learning-rate sweep (`10**linspace(-3, 0, 1000)`) to pick `lr = 0.1`, and scaled the model across three versions to compare on the dev set:

| Model | Embedding | Hidden | Params | Dev loss |
| :--- | :---: | :---: | :---: | :---: |
| v1 | 2 | 100 | 3,481 | 2.5015 |
| v2 | 10 | 250 | 14,797 | 2.1922 |
| v3 | 12 | 350 | 22,751 | 2.1796 |

Sampled names from the trained model and ran a final evaluation on the test set (`2.1786`) on v3, once.

**What broke / what confused me:**
- Didn't see why we use `10**lre` for the LR sweep, and thought `lrs` had duplicate values (`0.0010`) until `.tolist()` showed PyTorch was just rounding the printout.
- Wasn't sure how to trust a 1,000-step LR sweep with only one step per LR.
- Struggled with why `dev` and `test` need to be separate when weights are reset between runs.
- Didn't understand how `C` trains via backprop when `C[X]` is a lookup, or why similar letters cluster together.

**What I learned:**
- **LR sweep:** Exponential spacing tests every order of magnitude equally. Because adjacent LRs are nearly identical, the sweep acts as a stress test to find the fastest safe LR right before the loss spikes (`~0.1`).
- **Train / Dev / Test:** Train tunes weights, dev tunes hyperparameters, and test is a one-time unbiased check. Even when resetting weights, picking winners leaks info into your hyperparameters and burns the set.
- **Embeddings (`C`):** `C[X]` is a trainable lookup (equivalent to a first linear layer). Backprop sends gradients straight to the looked-up rows, and I think the shared `W1` pulls interchangeable letters to the same coordinates.
- **Scaling:** Bigger models lowered dev loss, but the gain shrank (2.5015 → 2.1922 → 2.1796). v3 also used a different lr schedule, so I can't say the size alone caused the last gain.

**Open questions:**
- What do we do next to get a better loss?
- How big is enough—or too big—in a neural network?

## 2026-11-22: makemore Part 3 (activations, gradients, BatchNorm)
**What I did:** Watched Karpathy's makemore Part 3 (1h55) and built
`04_makemore_batchnorm/04_makemore_batchnorm.ipynb`. Fixed the starting
loss (20.82 → 3.32; ideal 3.296) by shrinking the last layer (weights ×0.01,
bias 0), scaled the hidden weights with Kaiming init (gain 5/3 for tanh), and
added BatchNorm by hand: train 2.0678, val 2.1056 (Part 2 best dev: 2.1796).
Rebuilt it as Linear/BatchNorm1d/Tanh classes in a 5-layer network and read
four sensors: activation histograms, gradient histograms, weight gradient to
data ratio and update to data ratio. Compared a broken network (no BatchNorm,
weights ×10) with the healthy one.

**What broke / what confused me:**
- Why tanh near 0 is bad: it is almost a straight line, so stacked layers
  collapse into one linear layer.
- What the gain does, and what the four plots actually show and what I should
  aim for. At first I did not know what we were plotting.
- I did not keep the broken versions while following the video, so I rebuilt
  them afterwards from my own code.

**What I learned:** A too-high starting loss means the network starts
confidently wrong, because big last-layer weights give huge logits. Broken
network: 82–93% of the tanh outputs stuck at ±1, and gradients of 4.1e-2 in
the first layer against 1.8e-3 in the last. Healthy network: std 0.63–0.65,
only 1.7–2.8% stuck, gradients 2.6e-3 to 1.95e-3 in every layer. A stuck
neuron has gradient about 0, so it never learns. The gain cancels the
shrinking that tanh does at every layer (5/3 for tanh). BatchNorm forces each
layer's input to be well sized and uses whole-training-set statistics at test
time. The update to data ratio should sit near −3 on the log10 plot. Mine sat
at −2 to −2.5, a bit high, which is fine with BatchNorm. The way to read the
sensors is to check whether any layer looks different from the others.

**Open questions:** My sensor plots come from step 1000 only. Do the layers
still look the same after the full 200,000 steps? And why does BatchNorm make
the weight scale matter less?
