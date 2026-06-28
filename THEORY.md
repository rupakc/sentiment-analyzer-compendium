# Sentiment Analysis: Theory & Worked Examples

Companion to `RESEARCH.md`. That file surveys *what* the methods are; this one
derives *why they work* and walks a small worked example through each. Every
example uses tiny numbers you can check by hand.

Running toy corpus (used throughout for classical methods):

```
d1: "great movie"          → POS
d2: "great great film"     → POS
d3: "terrible movie"       → NEG
d4: "boring terrible film" → NEG
```

Vocabulary: {great, movie, film, terrible, boring}.

---

## 1. Lexicon scoring (VADER), worked

VADER doesn't just sum word scores — it applies ordered heuristics, then squashes
the total. Lexicon (illustrative valence scores, range −4…+4):

`good = +1.9`, `great = +3.1`, `not` = negator, `very` = booster (+0.293).

**Rule order:** (1) look up valence, (2) apply booster scaling, (3) apply negation
flip (×−0.74), (4) sum, (5) normalize.

**Example: "not very good"**
- `good` = +1.9
- booster `very` before it: `+1.9 + 0.293 = +2.193`
- negator `not` in the 3-word window: `2.193 × (−0.74) = −1.623`
- sum = −1.623

**Normalization** (the "compound" score) maps the raw sum `x` to (−1, 1):

```
compound = x / sqrt(x² + α),   α = 15
        = −1.623 / sqrt(1.623² + 15)
        = −1.623 / sqrt(17.63)
        = −0.386      → mildly negative ✓
```

The `α=15` constant controls how fast long, sentiment-heavy texts saturate toward
±1. Note the whole pipeline is deterministic and inspectable — that's the selling
point. The fragility is also visible: "not very good" being only *mildly* negative
is a hand-tuned guess, not learned.

---

## 2. Naïve Bayes, fully worked

**Model.** Pick the class maximizing the posterior:

```
ĉ = argmax_c  P(c) · ∏_i P(w_i | c)
```

"Naïve" = assume words are conditionally independent given the class (false, but
the *argmax* is often still right). Work in **log space** to avoid underflow:

```
log P(c|d) = log P(c) + Σ_i count(w_i) · log P(w_i | c)
```

**Multinomial NB with Laplace (add-1) smoothing:**

```
P(w | c) = (count(w in c) + 1) / (total tokens in c + |V|)
```

**Train on the toy corpus.** Priors: P(POS)=P(NEG)=½. |V| = 5.

POS tokens (d1,d2): great great great movie film → totals: great=3, movie=1,
film=1, terrible=0, boring=0. Total = 5.

NEG tokens (d3,d4): terrible terrible movie boring film → terrible=2, movie=1,
film=1, boring=1, great=0. Total = 5.

Smoothed likelihoods (denominator = 5 + 5 = 10):

| word | P(w\|POS) | P(w\|NEG) |
|------|-----------|-----------|
| great | (3+1)/10 = 0.40 | (0+1)/10 = 0.10 |
| terrible | (0+1)/10 = 0.10 | (2+1)/10 = 0.30 |
| movie | 0.20 | 0.20 |

**Classify "great movie":**

```
score_POS = log0.5 + log0.40 + log0.20 = −0.693 −0.916 −1.609 = −3.218
score_NEG = log0.5 + log0.10 + log0.20 = −0.693 −2.303 −1.609 = −4.605
```

POS wins (−3.218 > −4.605). The whole signal came from `great`; smoothing kept the
unseen-elsewhere words from zeroing the product. That add-1 step is *the* reason NB
doesn't fall over on sparse text.

---

## 3. Logistic Regression / MaxEnt

**Why discriminative beats NB here:** NB models `P(d|c)` (how documents are
*generated*) and inherits the false independence assumption into its
*probabilities*. LR models `P(c|d)` directly, so correlated features (e.g.
"new" + "york") don't double-count.

**Binary model.** With feature vector `x` (e.g. TF-IDF) and weights `w`:

```
P(y=1 | x) = σ(wᵀx + b),   σ(z) = 1 / (1 + e^−z)
```

Multiclass = softmax. "MaxEnt" is the same model derived from a different
principle: among all distributions matching the observed feature expectations,
pick the one with maximum entropy → provably the softmax/log-linear form.

**Training** minimizes cross-entropy (convex → global optimum):

```
L = −Σ_n [ y_n log p_n + (1−y_n) log(1−p_n) ] + λ‖w‖²
```

Gradient is clean: `∂L/∂w = Σ_n (p_n − y_n) x_n`. The `(prediction − truth)`
error term is the same shape that reappears in every neural net below — LR is the
single-neuron base case.

**Worked one step.** Feature = "contains `great`" (x=1), w=0, b=0, true y=1 (POS):
- p = σ(0) = 0.5, error = (0.5 − 1) = −0.5
- gradient = −0.5 × 1 = −0.5; with lr η=1: w ← 0 − 1·(−0.5) = **+0.5**

After the update `great` has positive weight → next prediction σ(0.5)=0.62, moving
toward 1. That's learning, one example at a time.

---

## 4. Linear SVM (the margin idea)

LR pushes *all* points away from the boundary; SVM only cares about the **closest**
ones (support vectors) and maximizes the **margin** `2/‖w‖`:

```
minimize  ½‖w‖² + C Σ ξ_n
s.t.      y_n(wᵀx_n + b) ≥ 1 − ξ_n,   ξ_n ≥ 0   (slack for non-separable data)
```

Equivalent unconstrained form = **hinge loss** `max(0, 1 − y·f(x))` + L2. Why it
dominated sparse text: the margin objective is robust in very high dimensions
(tens of thousands of n-gram features) where points are nearly always linearly
separable, and only support vectors matter so it's memory-lean. `C` trades margin
width vs. training errors.

---

## 5. HMM — forward, Viterbi, EM (worked)

A **generative** sequence model: hidden states `s` emit observed words `o`.
Parameters: start `π`, transitions `A[i→j]`, emissions `B[s→word]`.

Toy tagger, states {POS, NEG}:
- π = {POS: 0.5, NEG: 0.5}
- A = POS→POS 0.7, POS→NEG 0.3, NEG→NEG 0.7, NEG→POS 0.3
- B: P(great|POS)=0.6, P(great|NEG)=0.1, P(bad|POS)=0.1, P(bad|NEG)=0.6

**Viterbi for observation "great bad"** — best hidden path. δ = best prob to reach
a state at step t.

Step 1 (great): δ₁(POS)=0.5·0.6=0.30, δ₁(NEG)=0.5·0.1=0.05

Step 2 (bad):
```
δ₂(POS) = max(0.30·0.7, 0.05·0.3) · P(bad|POS) = max(0.21,0.015)·0.1 = 0.021  [from POS]
δ₂(NEG) = max(0.30·0.3, 0.05·0.7) · P(bad|NEG) = max(0.09,0.035)·0.6 = 0.054  [from POS]
```
Best final = NEG (0.054); backpointer → came from POS. **Decoded path: POS → NEG**
("great" then "bad"). The transition prior (POS likes to continue) lost to the
strong emission evidence for "bad". That tension is the whole model.

**Forward algorithm** is identical but sums instead of maxes → gives `P(observations)`,
used for training. **Baum-Welch (EM)** learns A, B, π without labeled states:
E-step computes expected state/transition counts via forward-backward; M-step
re-normalizes them into probabilities; repeat until likelihood converges. The
catch: EM only finds a *local* optimum, and the generative independence assumptions
keep HMMs from using rich features — which is exactly what CRFs fix.

---

## 6. CRF — the discriminative sequence model

**The label-bias problem CRFs solve.** HMMs/MEMMs normalize *per state* (locally),
so a state with few outgoing transitions passes probability through almost
regardless of the observation — it can ignore evidence. CRFs normalize over the
**whole sequence** (globally), so every observation gets to vote. This is *the*
reason CRFs beat HMMs/MEMMs on tagging.

**Model** (linear-chain):

```
P(y | x) = (1/Z(x)) · exp( Σ_t Σ_k λ_k f_k(y_{t−1}, y_t, x, t) )

Z(x) = Σ_{y'} exp( Σ_t Σ_k λ_k f_k(y'_{t−1}, y'_t, x, t) )   ← global partition fn
```

**Feature functions `f_k`** are arbitrary, possibly overlapping, real/boolean
functions — this freedom is the payoff vs. HMM. For aspect-term extraction with
BIO tags, typical features at position t:

```
f1 = 1 if (word_t is in sentiment-lexicon AND y_t = B-ASPECT)
f2 = 1 if (word_t is capitalized AND y_t = B-ASPECT)
f3 = 1 if (POS(word_t) = NOUN AND y_t ∈ {B,I}-ASPECT)
f4 = 1 if (y_{t−1} = B-ASPECT AND y_t = I-ASPECT)     ← transition feature
f5 = 1 if (word_{t−1} = "the" AND y_t = B-ASPECT)     ← uses neighboring words freely
```

An HMM *cannot* express f2, f3, f5 cleanly because it would have to model the joint
distribution of word+caps+POS+context. CRF just adds them as features and learns a
weight `λ_k` each.

**Training:** maximize log-likelihood `Σ log P(y|x) − regularizer`; convex; gradient
of each `λ_k` = (empirical feature count) − (model-expected feature count, via
forward-backward over Z). **Decoding:** Viterbi, same as HMM. Pair with a BiLSTM
that produces the features and you get the classic **BiLSTM-CRF**.

---

## 7. Word embeddings (word2vec skip-gram)

Goal: a word's vector should predict its neighbors. Skip-gram objective for center
word `c`, context word `o`:

```
P(o | c) = softmax(u_oᵀ v_c) = exp(u_oᵀ v_c) / Σ_w exp(u_wᵀ v_c)
```

The full softmax sums over the whole vocabulary (millions) — too slow. **Negative
sampling** replaces it: make the real (c,o) pair score high and `k` random "noise"
pairs score low:

```
log σ(u_oᵀ v_c) + Σ_{j=1..k} E_{w~Pn} [ log σ(−u_wᵀ v_c) ]
```

Result: vectors where cosine similarity ≈ semantic similarity, and the famous
`king − man + woman ≈ queen` geometry. **Sentiment-relevant limitation:** "good"
and "bad" appear in identical contexts ("the movie was ___"), so they land
*near each other* — pure word2vec can be sentiment-blind. Fixes: retrofitting with
sentiment lexicons, or just let a downstream supervised layer reweight them.
**fastText** adds character n-grams so unseen/misspelled words ("amazinggg") still
get a vector.

---

## 8. LSTM (the gate equations, intuited)

A vanilla RNN's gradient through `t` steps multiplies ~`W` by itself `t` times →
vanishes or explodes. The LSTM adds a **cell state** `C_t` with an additive path,
so gradients flow without repeated multiplication. At each step (σ = sigmoid gate
in [0,1], ⊙ = elementwise):

```
f_t = σ(W_f·[h_{t−1}, x_t] + b_f)        forget: what to drop from memory
i_t = σ(W_i·[h_{t−1}, x_t] + b_i)        input:  what new info to consider
g_t = tanh(W_g·[h_{t−1}, x_t] + b_g)     candidate memory content
C_t = f_t ⊙ C_{t−1} + i_t ⊙ g_t          update cell (the additive highway)
o_t = σ(W_o·[h_{t−1}, x_t] + b_o)        output gate
h_t = o_t ⊙ tanh(C_t)                    hidden output
```

**Sentiment intuition:** reading "not bad", the gates can store a "negation
pending" signal in `C_t` and apply it when "bad" arrives — exactly the long-range
negation-scope handling that BoW models can't do. **GRU** merges forget+input into
one gate (fewer params, similar performance). **BiLSTM** runs two of these in
opposite directions so each token's representation sees both left and right context.

---

## 9. Attention & Transformers (worked)

**Scaled dot-product attention.** Every token gets a Query, Key, Value vector.
Token i's output = weighted average of all Values, weighted by how well its Query
matches each Key:

```
Attention(Q,K,V) = softmax( QKᵀ / √d_k ) V
```

`√d_k` keeps the dot products from growing with dimension and saturating softmax.

**Tiny worked example.** Sentence "not good", d_k = 1 (scalars for hand-math).
Suppose for the query of "good": its dot-products with keys are
`score(good→not) = 2.0`, `score(good→good) = 1.0`.

```
softmax([2.0, 1.0]) = [e²/(e²+e¹), e¹/(e²+e¹)] = [7.39/10.10, 2.72/10.10]
                    = [0.731, 0.269]
```

So "good"'s new representation is `0.731·V(not) + 0.269·V(good)` — it has pulled in
73% of the "not" signal. *That* is how a transformer flips "good" to negative: the
representation of "good" literally absorbs "not". No fixed window, no recurrence —
it's direct, learned, and parallel across all tokens at once.

**Multi-head** runs h such attentions with different learned Q/K/V projections and
concatenates — different heads specialize (one tracks negation, one tracks the
subject, etc.). A Transformer block = multi-head self-attention + feed-forward,
each wrapped in residual + layer-norm, stacked N times.

---

## 10. BERT fine-tuning for sentiment

**Pretraining** (self-supervised, no labels): Masked LM (predict 15% randomly
masked tokens) forces bidirectional context understanding. This is where the model
learns that "sick" means different things in "sick beat" vs "feeling sick" —
contextual embeddings.

**Fine-tuning for classification:**
1. Prepend a special `[CLS]` token; its final-layer vector `h_[CLS] ∈ ℝ⁷⁶⁸` is a
   pooled sentence representation.
2. Add one linear layer: `logits = W·h_[CLS] + b`, then softmax.
3. Train end-to-end on the labeled sentiment set with a *small* learning rate
   (~2e-5) for 2–4 epochs — you're nudging pretrained weights, not learning from
   scratch, which is why a few thousand labels suffice.

**Why it wins:** the heavy lifting (syntax, word sense, context) is already in the
pretrained weights; the sentiment task only has to learn a thin readout. Same
reason DistilBERT keeps ~97% of accuracy at 40% the size — most of the knowledge
is robust to compression.

---

## Through-line

Watch one quantity travel across all ten sections: the way a model represents
**"not"**.

| Method | How "not good" becomes negative |
|--------|-------------------------------|
| Lexicon | hand rule: ×−0.74 within a window |
| Naïve Bayes | only if "not_good" bigram was a feature |
| LR/SVM | learned weight on a "not_good" feature |
| HMM | transition + emission probabilities, weakly |
| CRF | a feature function over (prev word, tag) |
| LSTM | a "negation pending" bit stored in the cell state |
| Transformer | "good"'s vector absorbs 73% of "not" via attention |
| BERT | all of the above, pretrained, then a thin readout |

Every advance is the same move: take something previously **hand-specified** and
let the model **learn it from data** — paying in compute and interpretability for
generality. That single sentence is the 20-year story.
