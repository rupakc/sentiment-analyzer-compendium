# Sentiment Analysis: A Compendium of Techniques (1990s → Today)

A practitioner's survey of the major families of sentiment analysis methods, in
roughly the order they became dominant. For each: the core idea, the math, what
it's good at, where it breaks, and what it costs to run. Ends with a
recommendation for the app.

---

## 0. Framing the problem

**Sentiment analysis** = mapping text → an opinion label. The "label" varies:

| Task | Output | Example |
|------|--------|---------|
| Polarity classification | pos / neg / (neutral) | "I love it" → pos |
| Fine-grained / ordinal | 1–5 stars | review → 4 |
| Aspect-based (ABSA) | (aspect, polarity) pairs | "battery great, screen bad" → (battery, pos), (screen, neg) |
| Emotion detection | joy/anger/fear/… | multi-label |
| Subjectivity | objective vs. subjective | filter facts from opinions |
| Intensity / VAD | continuous valence-arousal-dominance | regression |

Two recurring hard problems thread through every method below:

1. **Negation & scope** — "not good", "I wouldn't say it's bad".
2. **Context / world knowledge** — sarcasm, irony, comparatives ("better than X"),
   domain-flipped words ("unpredictable" is good for a movie, bad for a car).

The history is largely a story of methods getting progressively better at #2 by
learning representations instead of hand-coding rules.

---

## 1. Rule-based & lexicon methods (≈1997–2010, still in production)

### 1.1 Core idea
Maintain a **sentiment lexicon**: a dictionary mapping words → polarity scores.
Score a document by aggregating (sum / average) the scores of words it contains,
modified by hand-written rules for negation, intensifiers, and punctuation.

### 1.2 The major lexicons
- **General Inquirer** (1966, Harvard) — the granddaddy; tagged word categories.
- **SentiWordNet** — assigns pos/neg/objective scores to WordNet synsets.
- **MPQA Subjectivity Lexicon** — strong/weak subjective clues.
- **Bing Liu's Opinion Lexicon** — ~6,800 pos/neg words.
- **AFINN** — words scored −5…+5 (compact, fast).
- **VADER** (2014) — purpose-built for social media; handles emoji, slang,
  ALL-CAPS, "!!!", and degree modifiers. Still an excellent baseline today.

### 1.3 The rules that matter
- **Negation**: flip or dampen polarity within a window after a negator
  ("not", "never", "n't"). Scope detection is the hard part.
- **Intensifiers/downtoners**: "very" amplifies, "slightly" dampens.
- **Contrast**: "but" — clause after "but" usually dominates.
- **Punctuation/caps**: "!!!", "GREAT" boost intensity (VADER's contribution).

### 1.4 Strengths / weaknesses
- ✅ Zero training data, fully interpretable, instant, trivially debuggable.
- ✅ Strong, honest baseline — often within 10–15 pts of fancy models on clean data.
- ❌ Coverage gaps (OOV words, new slang), no real context, sarcasm-blind,
  domain drift requires manual lexicon edits.

**When to use today:** cold start, no labels, need explainability, social-media
firehose where VADER is hard to beat for the cost.

---

## 2. Classical machine learning (≈2002–2014)

The seminal moment: **Pang, Lee & Vaithyanathan (2002)** showed supervised
classifiers on bag-of-words beat lexicon methods on movie reviews — and noted
sentiment is "harder than topic classification."

### 2.1 Feature engineering (where the real work was)
- **Bag-of-words / n-grams** — unigrams + bigrams capture "not good".
- **TF-IDF weighting** — downweight ubiquitous terms.
- **Booleans vs. counts** — *presence* of a word often beat *frequency* for sentiment.
- Negation tagging (append `_NEG` to tokens after a negator until punctuation),
  POS tags, lexicon-derived features, emoticons.

### 2.2 The workhorse classifiers
- **Naïve Bayes** — `P(c|d) ∝ P(c)·∏ P(wᵢ|c)`. Fast, surprisingly strong with
  good smoothing; **Multinomial NB** and **NBSVM** (NB features + SVM) are great baselines.
- **Logistic Regression / MaxEnt** — discriminative, no independence assumption,
  well-calibrated probabilities. Often the best classical choice.
- **Linear SVM** — maximum-margin; historically the top performer on high-dim
  sparse text. `LinearSVC` / `liblinear` scales fine.
- Also-rans: decision trees, kNN, ensembles (rarely worth it on sparse text).

### 2.3 Strengths / weaknesses
- ✅ Great accuracy-per-compute, fast inference, modest data needs (thousands of labels).
- ✅ Still the right tool for many production classifiers.
- ❌ BoW discards word order beyond n-grams; feature engineering is manual labor;
  doesn't generalize across domains without retraining.

---

## 3. Sequence models: HMMs and CRFs (≈2004–2015)

These shine when sentiment is a **labeling-over-tokens** problem, not a
single-document label — i.e. **aspect/target extraction** and fine-grained ABSA.

### 3.1 Hidden Markov Models (HMM)
- **Generative**: models `P(observations, hidden states)` via transition matrix
  `A` (state→state) and emission `B` (state→word), trained with Baum-Welch (EM),
  decoded with **Viterbi**.
- In sentiment: hidden states ≈ sentiment/aspect tags; less common than CRFs
  because the independence assumptions are too strong for text features.

### 3.2 Conditional Random Fields (CRF)
- **Discriminative** sequence model — the dominant choice for sequence labeling
  before BiLSTMs. Models `P(label_sequence | tokens)` directly:

  `P(y|x) = (1/Z(x)) · exp( Σₜ Σₖ λₖ fₖ(y_{t-1}, yₜ, x, t) )`

- **Why it beat HMMs:** lets you pour in arbitrary, overlapping features
  (word, prefix/suffix, POS, capitalization, lexicon hit, surrounding window)
  without modeling their joint distribution. Globally normalized → no label bias.
- **Use in sentiment:** opinion-target extraction, aspect-term extraction
  (BIO tagging), sentiment-expression boundaries. `sklearn-crfsuite` is the
  classic toolkit. Still competitive when paired with neural features (BiLSTM-CRF).

### 3.3 Trade-off
- ✅ Best non-neural approach for *structured* sentiment (find what's being talked
  about + its polarity), interpretable feature weights.
- ❌ Feature engineering heavy; training slower than NB/LR; for whole-document
  polarity it's overkill.

---

## 4. Deep learning, pre-transformer (≈2013–2018)

The shift: **learned dense representations** replace hand-built features.

### 4.1 Word embeddings (the enabler)
- **word2vec** (2013), **GloVe**, **fastText** (subword → handles OOV).
- Map words → dense vectors where geometry ≈ meaning. Sentiment models now start
  from these instead of one-hot BoW. Limitation: *static* — "bank" has one vector.

### 4.2 The architectures
- **CNN for text** (Kim, 2014) — 1-D convolutions over embeddings act as learned
  n-gram detectors + max-pooling. Fast, strong, embarrassingly parallel.
- **RNN / LSTM / GRU** — process tokens sequentially with memory; LSTM/GRU gates
  fix vanishing gradients. Naturally model order and long-ish dependencies.
- **BiLSTM** — read left→right and right→left, concatenate. The standard sentence
  encoder of the era; **BiLSTM-CRF** was state-of-the-art for sequence labeling/ABSA.
- **Attention** (Bahdanau, 2015) — let the model weight which tokens matter;
  also yields a crude interpretability heat-map. Tree-LSTMs exploited parse structure.

### 4.3 Trade-off
- ✅ Big jump on context/negation/long-range; less manual feature work.
- ❌ Needs more data + GPU; static embeddings still miss word-sense; sequential
  RNNs are slow to train.

---

## 5. Transformers & pretrained LMs (2018 → today, state of the art)

### 5.1 The idea
**Self-attention** (Vaswani et al., "Attention Is All You Need", 2017) replaces
recurrence — every token attends to every other in parallel. Combined with
**pretrain-then-finetune**: train a huge LM on raw text, then fine-tune on a small
labeled sentiment set.

- **BERT** (2018) — bidirectional, contextual embeddings; fine-tune by adding a
  classification head on the `[CLS]` token. Crushed prior benchmarks.
- **Variants worth knowing:** RoBERTa (better-trained BERT), DistilBERT (40%
  smaller, ~97% of accuracy — great for prod), DeBERTa (often top of leaderboards),
  domain-tuned ones like **BERTweet** / **twitter-roberta** for social text.
- **Contextual embeddings** finally solve word-sense: "sick" in "sick beat" vs.
  "feeling sick" get different vectors.

### 5.2 Large Language Models (2022 → )
- **Zero/few-shot** sentiment via prompting — no training data, handles nuance,
  sarcasm, and aspect-based extraction in one shot, returns structured JSON.
- ✅ Best accuracy on hard/nuanced text, no labels needed, flexible output schema,
  explanations for free. For nuanced ABSA, a modern Claude model is the strongest
  off-the-shelf option.
- ❌ Latency + $/call + rate limits; overkill for high-volume simple polarity;
  needs guardrails for consistency.

### 5.3 Trade-off summary
- ✅ State of the art on essentially every benchmark; minimal/zero feature work.
- ❌ Compute-heavy (fine-tuned transformers need GPU; LLM APIs cost per call);
  less interpretable; latency.

---

## 6. Cross-cutting concerns

### 6.1 Evaluation
- **Metrics:** accuracy (only if balanced), **macro-F1** (preferred for skew),
  precision/recall per class, MAE/QWK for ordinal star ratings.
- **Always** report a lexicon (VADER) and a LR/NB baseline before claiming a win.
- Watch **domain shift** — a model trained on movie reviews tanks on tweets.

### 6.2 Canonical datasets
- IMDb (50k movie reviews, binary) · SST / SST-2 (Stanford, fine-grained, phrase-level)
- Amazon / Yelp reviews (stars) · SemEval Twitter tasks (2013–2017) · SemEval ABSA
  (2014–2016) · Sentiment140 (1.6M weakly-labeled tweets).

### 6.3 Perennial hard cases (test set should include all of these)
Negation scope · sarcasm/irony · comparatives · mixed/aspect-conflicting reviews ·
domain-dependent words · neutral vs. weak sentiment · code-switching & emoji.

---

## 7. The arc, in one table

| Era | Family | Repr. | Data need | Compute | Interpretable | Ceiling |
|-----|--------|-------|-----------|---------|---------------|---------|
| ~2000 | Lexicon/rules | hand dict | none | trivial | ★★★ | low |
| 2002+ | NB/LR/SVM + n-grams | sparse BoW | 1k–10k | low | ★★ | medium |
| 2004+ | HMM/CRF | features | 1k–10k | med | ★★ | medium (structured) |
| 2014+ | CNN/BiLSTM(+CRF) | embeddings | 10k+ | GPU | ★ | high |
| 2018+ | BERT family | contextual | 100s–1k finetune | GPU | low | very high |
| 2022+ | LLM prompting | — | 0 | API $ | low-med | very high (nuance) |

**Takeaway:** progress = moving feature-discovery from human → model, at the cost
of compute and interpretability. None of these is obsolete — the right pick is a
function of labels available, latency/cost budget, need for explanation, and
whether you need document-level polarity or structured aspect extraction.

---

## 8. Recommendation for the app

The pedagogical value here is *comparison*. Build an app that runs the **same input
text through one representative of each family side-by-side** and shows their
verdicts + confidence + (where possible) which words drove the decision:

1. **Lexicon** — VADER (zero deps, instant, shows per-word contributions).
2. **Classical ML** — Logistic Regression / NBSVM on TF-IDF (train on IMDb or SST-2).
3. **Sequence** — a CRF demo on aspect-term extraction (shows the *structured* task).
4. **Deep learning** — a fine-tuned DistilBERT/twitter-roberta from HF.
5. **LLM** — a Claude call for nuanced + aspect-based + explanation.

This makes the trade-offs (speed vs. nuance, interpretability vs. accuracy)
*visible and tangible* — which is the whole point of a 20-year retrospective.

> ponytail note: don't build all 5 model integrations up front. Start with #1 (VADER)
> + #5 (LLM) — the two extremes of the spectrum, both zero-training — wire up the
> side-by-side UI, and add #2–#4 only once the comparison framing proves useful.
