export interface MathBlock {
  /** short label / name for the equation, shown in mono */
  label: string;
  /** LaTeX, rendered in display mode */
  tex: string;
  /** one-line plain-language gloss */
  note?: string;
}

export interface ProofStep {
  /** optional LaTeX for this line of the derivation */
  tex?: string;
  /** plain-language explanation of the step */
  text: string;
}

export interface Proof {
  claim: string;
  steps: ProofStep[];
}

export interface Guide {
  id: string;
  title: string;
  era: string;
  tagline: string;
  intuition: string;
  howItWorks: string[];
  math: MathBlock[];
  proof?: Proof;
  /** hand-checkable example, shown in monospace */
  workedExample: string;
  pros: string[];
  cons: string[];
  whenToUse: string;
}

export const GUIDES: Guide[] = [
  // ───────────────────────────────────────────────────────────── 1. LEXICON
  {
    id: "lexicon",
    title: "Lexicon / Rule-based (VADER)",
    era: "≈1997–2010",
    tagline: "A dictionary of feelings, summed by hand-written rules.",
    intuition:
      "Keep a dictionary that maps words to polarity scores, add up the scores of the words a sentence contains, and patch the obvious failure modes — negation, intensifiers, punctuation — with ordered heuristics. No training data, no model to fit: the knowledge lives entirely in the word list and the rules. Everything it does is inspectable, which is both its charm and its ceiling.",
    howItWorks: [
      "Look up each token's valence in the dictionary (VADER scores words roughly −4…+4, purpose-built for social media: it knows emoji, slang, and ALL-CAPS).",
      "Apply degree modifiers: a booster like 'very' before a word amplifies it; a downtoner like 'slightly' dampens it.",
      "Apply negation: a negator ('not', 'never', \"n't\") within a small window flips the sign and scales by ≈ −0.74 (a flip plus a dampening, since negation rarely fully inverts intensity).",
      "Sum the adjusted valences into a raw score x, then squash x into (−1, 1) with a smooth normalization to get the final 'compound' score.",
    ],
    math: [
      {
        label: "compound normalization",
        tex: "\\text{compound} = \\dfrac{x}{\\sqrt{x^{2} + \\alpha}}, \\qquad \\alpha = 15",
        note: "Maps the unbounded raw sum x into (−1, 1); α controls how fast sentiment-heavy text saturates toward ±1.",
      },
      {
        label: "negation rule",
        tex: "v' = v \\times (-0.74)",
        note: "A negator in the window flips and dampens — 'not great' is negative but not as negative as 'terrible'.",
      },
      {
        label: "booster rule",
        tex: "v' = v + b, \\qquad b_{\\text{very}} = +0.293",
        note: "Intensifiers add a fixed increment before negation and summation are applied.",
      },
    ],
    workedExample: [
      'Input: "not very good"',
      "  good                        = +1.900",
      "  + booster 'very'  (+0.293)  = +2.193",
      "  × negator 'not'   (−0.74)   = −1.623   (raw sum x)",
      "",
      "  compound = x / sqrt(x^2 + 15)",
      "           = −1.623 / sqrt(1.623^2 + 15)",
      "           = −1.623 / sqrt(17.63)",
      "           = −0.386   →  mildly negative ✓",
    ].join("\n"),
    pros: [
      "Zero training data — works on day one, in any domain.",
      "Fully interpretable: you can read off exactly which word and rule produced the score.",
      "Trivially fast and cheap; trivial to debug and patch.",
      "An honest, hard-to-beat baseline on clean social-media text (VADER especially).",
    ],
    cons: [
      "Coverage gaps: out-of-vocabulary words and new slang score zero.",
      "No real context — sarcasm-blind, weak on long-range negation scope.",
      "Domain drift ('unpredictable' is good for a film, bad for a car) needs manual dictionary edits.",
      "Constants like −0.74 and α=15 are hand-tuned guesses, not learned from data.",
    ],
    whenToUse:
      "Cold start with no labels, a hard requirement for explainability, or a high-volume social-media firehose where VADER's accuracy-per-dollar is hard to beat.",
  },

  // ──────────────────────────────────────────────────────── 2. CLASSICAL ML
  {
    id: "classical",
    title: "Classical ML (Logistic Regression)",
    era: "2002+",
    tagline: "Turn text into sparse vectors, then learn the weights from labels.",
    intuition:
      "Pang, Lee & Vaithyanathan (2002) showed that a supervised classifier on bag-of-words beats dictionary methods on movie reviews. The recipe: represent a document as a sparse vector of (TF-IDF-weighted) n-gram counts, then fit a linear model. Naïve Bayes models how documents are generated, P(d|c); Logistic Regression models the answer directly, P(c|d) — which is why correlated features like 'new'+'york' don't get double-counted. LR is the single-neuron base case of every neural net that follows.",
    howItWorks: [
      "Tokenize into unigrams + bigrams (bigrams let 'not good' be a single feature) and weight by TF-IDF so ubiquitous terms count for less.",
      "Naïve Bayes: estimate P(word|class) with Laplace smoothing and pick the class with the highest log-posterior.",
      "Logistic Regression: pass a weighted sum through the sigmoid to get P(class=1|x), and fit the weights by minimizing cross-entropy (a convex objective → a single global optimum).",
      "Linear SVM: instead of probabilities, maximize the margin to the nearest points — historically the top performer on high-dimensional sparse text.",
    ],
    math: [
      {
        label: "Naïve Bayes argmax",
        tex: "\\hat{c} = \\arg\\max_{c}\\; \\log P(c) + \\sum_{i} \\text{count}(w_i)\\,\\log P(w_i \\mid c)",
        note: "Work in log-space to avoid underflow; the naïve independence assumption is false but the argmax is often still right.",
      },
      {
        label: "Laplace smoothing",
        tex: "P(w \\mid c) = \\dfrac{\\text{count}(w, c) + 1}{\\big(\\sum_{w'} \\text{count}(w', c)\\big) + |V|}",
        note: "Add-1 stops an unseen word from zeroing the whole product — the reason NB survives sparse text.",
      },
      {
        label: "TF-IDF weight",
        tex: "\\text{tfidf}(w, d) = \\text{tf}(w, d)\\cdot \\log\\dfrac{N}{\\text{df}(w)}",
        note: "Down-weights terms that appear in many documents; rare-but-present words carry the signal.",
      },
      {
        label: "logistic model",
        tex: "P(y{=}1 \\mid x) = \\sigma(w^{\\top}x + b), \\qquad \\sigma(z) = \\dfrac{1}{1 + e^{-z}}",
        note: "The sigmoid squashes the linear score into a calibrated probability.",
      },
      {
        label: "cross-entropy loss",
        tex: "\\mathcal{L} = -\\sum_{n}\\big[\\, y_n \\log p_n + (1 - y_n)\\log(1 - p_n)\\,\\big] + \\lambda\\lVert w\\rVert^{2}",
        note: "Convex in w; the λ term is L2 regularization to curb overfitting on tens of thousands of features.",
      },
      {
        label: "SVM margin / hinge loss",
        tex: "\\min_{w,b}\\; \\tfrac{1}{2}\\lVert w\\rVert^{2} + C\\sum_n \\max\\!\\big(0,\\, 1 - y_n(w^{\\top}x_n + b)\\big)",
        note: "Maximize the margin 2/‖w‖; only the closest points (support vectors) matter, so it is memory-lean and robust in high dimensions.",
      },
    ],
    proof: {
      claim:
        "The gradient of the logistic cross-entropy loss is the clean 'error × input' form ∂L/∂wⱼ = Σₙ (pₙ − yₙ) xₙⱼ — the same shape that reappears in every neural network.",
      steps: [
        {
          text: "Take one example. Let z = wᵀx + b and p = σ(z). The per-example loss is:",
          tex: "\\ell = -\\big[\\, y\\log p + (1 - y)\\log(1 - p)\\,\\big]",
        },
        {
          text: "First, the derivative of the loss w.r.t. the probability p:",
          tex: "\\dfrac{\\partial \\ell}{\\partial p} = -\\dfrac{y}{p} + \\dfrac{1 - y}{1 - p} = \\dfrac{p - y}{p(1 - p)}",
        },
        {
          text: "The sigmoid has the famously tidy derivative σ′(z) = σ(z)(1 − σ(z)) = p(1 − p):",
          tex: "\\dfrac{\\partial p}{\\partial z} = p(1 - p)",
        },
        {
          text: "Chain the two: the p(1 − p) terms cancel exactly, leaving just the prediction error:",
          tex: "\\dfrac{\\partial \\ell}{\\partial z} = \\dfrac{p - y}{p(1 - p)} \\cdot p(1 - p) = p - y",
        },
        {
          text: "Finally z is linear in w, so ∂z/∂wⱼ = xⱼ. Summing over the dataset:",
          tex: "\\dfrac{\\partial \\mathcal{L}}{\\partial w_j} = \\sum_n (p_n - y_n)\\,x_{nj}\\quad\\blacksquare",
        },
        {
          text: "Interpretation: each feature's weight moves in proportion to how wrong the prediction was, scaled by how active that feature is. Gradient descent on this convex surface lands on the global optimum.",
        },
      ],
    },
    workedExample: [
      "Toy corpus, P(POS)=P(NEG)=0.5, |V|=5.  Classify 'great movie':",
      "",
      "  P(great|POS)=0.40   P(movie|POS)=0.20",
      "  P(great|NEG)=0.10   P(movie|NEG)=0.20",
      "",
      "  score_POS = log0.5 + log0.40 + log0.20 = −3.218",
      "  score_NEG = log0.5 + log0.10 + log0.20 = −4.605",
      "  → POS wins (−3.218 > −4.605); the signal is all 'great'.",
      "",
      "LR, one SGD step on feature 'great' (x=1, w=0, true y=1):",
      "  p = σ(0) = 0.5,  error = p − y = −0.5",
      "  w ← 0 − η·(−0.5) = +0.5   (η=1)   → next pred σ(0.5)=0.62 ✓",
    ].join("\n"),
    pros: [
      "Excellent accuracy-per-compute and millisecond inference.",
      "Modest data needs (a few thousand labels) and well-calibrated probabilities (LR).",
      "Convex training → reproducible global optimum, no random seeds to babysit.",
      "Still the right production tool for many straightforward polarity classifiers.",
    ],
    cons: [
      "Bag-of-words discards word order beyond the n-grams you explicitly add.",
      "Feature engineering (negation tagging, n-gram choice) is manual labor.",
      "Doesn't transfer across domains without retraining.",
      "Naïve Bayes's independence assumption mis-calibrates its probabilities (the argmax usually survives).",
    ],
    whenToUse:
      "You have a few thousand labeled examples in-domain, need fast and cheap inference at scale, and want calibrated, debuggable predictions. The default classical baseline before reaching for anything neural.",
  },

  // ───────────────────────────────────────────────────── 3. SEQUENCE MODELS
  {
    id: "sequence",
    title: "Sequence models (HMM & CRF)",
    era: "2004–2015",
    tagline: "Label every token, not the whole document — for aspect extraction.",
    intuition:
      "Some sentiment tasks are labeling-over-tokens, not one label per document: find which spans name an aspect ('battery', 'screen') and tag them BIO. HMMs are generative — hidden sentiment/aspect states emit words — and decode the best path with Viterbi. CRFs are discriminative: they score a whole label sequence at once and normalize globally, which lets them use arbitrary overlapping features and sidesteps the label-bias problem that cripples locally-normalized models.",
    howItWorks: [
      "HMM: parameters are start π, transition A[i→j], and emission B[state→word]; Baum-Welch (EM) learns them unsupervised, Viterbi decodes the most likely state path.",
      "Viterbi keeps δₜ(j), the best probability of any path that ends in state j at step t, and a backpointer; it is dynamic programming over the trellis.",
      "CRF: define feature functions fₖ over (previous tag, current tag, the whole observation, position) — they can freely look at capitalization, POS, dictionary hits, neighboring words.",
      "CRF training maximizes the (convex) conditional log-likelihood; the global partition function Z(x) sums over all label sequences via forward-backward; decoding is Viterbi again.",
    ],
    math: [
      {
        label: "HMM Viterbi recursion",
        tex: "\\delta_t(j) = \\Big(\\max_{i}\\; \\delta_{t-1}(i)\\,a_{ij}\\Big)\\, b_j(o_t)",
        note: "Best path probability into state j at time t = best predecessor × transition × emission of the current word.",
      },
      {
        label: "HMM forward",
        tex: "\\alpha_t(j) = \\Big(\\sum_{i}\\alpha_{t-1}(i)\\,a_{ij}\\Big)\\,b_j(o_t)",
        note: "Identical to Viterbi but sums instead of maxes → gives P(observations), used for training.",
      },
      {
        label: "linear-chain CRF",
        tex: "P(y \\mid x) = \\dfrac{1}{Z(x)}\\exp\\!\\Big(\\sum_{t}\\sum_{k}\\lambda_k\\, f_k(y_{t-1}, y_t, x, t)\\Big)",
        note: "Score the whole sequence as a sum of weighted features; learn one λₖ per feature.",
      },
      {
        label: "global partition function",
        tex: "Z(x) = \\sum_{y'}\\exp\\!\\Big(\\sum_{t}\\sum_{k}\\lambda_k\\, f_k(y'_{t-1}, y'_t, x, t)\\Big)",
        note: "Normalizes over every possible label sequence — the source of CRF's edge over locally-normalized models.",
      },
    ],
    proof: {
      claim:
        "Global normalization (CRF) avoids the label-bias problem that afflicts locally-normalized models (MEMM/HMM-style).",
      steps: [
        {
          text: "A locally-normalized model factors as a product of per-state transition distributions, each summing to 1 on its own:",
          tex: "P(y \\mid x) = \\prod_{t} P(y_t \\mid y_{t-1}, x_t), \\qquad \\sum_{y_t} P(y_t \\mid y_{t-1}, x_t) = 1",
        },
        {
          text: "Consequence: a state with only one outgoing transition must send probability 1 along it — no matter what the observation xₜ says. The observation is effectively ignored.",
        },
        {
          text: "So a path through low-entropy (few-branch) states wins on structure alone; the model is biased toward states with fewer successors, regardless of the evidence. That is the label-bias problem.",
        },
        {
          text: "A CRF instead normalizes once, globally, by Z(x). No per-state probability is forced to sum to 1, so each feature — including every emission-like feature — gets to vote on the entire sequence:",
          tex: "P(y \\mid x) = \\dfrac{1}{Z(x)}\\exp\\!\\Big(\\sum_{t}\\sum_{k}\\lambda_k\\, f_k(y_{t-1}, y_t, x, t)\\Big)",
        },
        {
          text: "Because Z(x) couples all positions, a strong observation late in the sentence can outweigh a structurally 'cheap' path — exactly what local normalization cannot do. This is why CRFs beat HMMs/MEMMs on tagging. ∎",
        },
      ],
    },
    workedExample: [
      "HMM Viterbi for 'great bad'.  States {POS, NEG}.",
      "  π = {POS:0.5, NEG:0.5}",
      "  A: POS→POS 0.7, POS→NEG 0.3, NEG→NEG 0.7, NEG→POS 0.3",
      "  B: P(great|POS)=0.6, P(great|NEG)=0.1, P(bad|POS)=0.1, P(bad|NEG)=0.6",
      "",
      "  Step 1 (great): δ(POS)=0.5·0.6=0.30,  δ(NEG)=0.5·0.1=0.05",
      "  Step 2 (bad):",
      "    δ(POS)=max(0.30·0.7, 0.05·0.3)·0.1 = 0.021  [from POS]",
      "    δ(NEG)=max(0.30·0.3, 0.05·0.7)·0.6 = 0.054  [from POS]",
      "  Best final = NEG (0.054).  Decoded path: POS → NEG ✓",
      "  (strong 'bad' emission beat the POS→POS transition prior)",
    ].join("\n"),
    pros: [
      "The best non-neural approach for structured sentiment (what is being talked about + its polarity).",
      "CRFs admit arbitrary overlapping features (POS, caps, dictionary, window) without modeling their joint distribution.",
      "Interpretable learned feature weights; convex CRF training.",
      "Pairs cleanly with neural features as the classic BiLSTM-CRF.",
    ],
    cons: [
      "Feature engineering is heavy; training is slower than NB/LR.",
      "HMM's generative independence assumptions are too strong for rich text features.",
      "EM for HMMs only finds a local optimum.",
      "Overkill for plain document-level polarity.",
    ],
    whenToUse:
      "Aspect-term / opinion-target extraction and other BIO-tagging tasks where you need the spans, not just one label. Reach for a CRF (or BiLSTM-CRF) over an HMM whenever you want rich, overlapping features.",
  },

  // ──────────────────────────────────────────────────────── 4. DEEP LEARNING
  {
    id: "deep",
    title: "Deep learning (CNN · LSTM · Transformers/DistilBERT)",
    era: "2013+",
    tagline: "Learn dense representations instead of hand-building features.",
    intuition:
      "Replace one-hot bag-of-words with dense embeddings where geometry ≈ meaning, then learn the features. CNNs slide filters over embeddings as learned n-gram detectors. LSTMs add a cell-state 'highway' so gradients survive long sequences and the network can stash a 'negation pending' bit until the word it modifies arrives. Transformers drop recurrence entirely: self-attention lets every token directly absorb information from every other token in parallel — and BERT pretrains this on raw text so fine-tuning needs only a thin readout.",
    howItWorks: [
      "Word2vec/GloVe/fastText map words to dense vectors trained so a word predicts its neighbors; fastText adds character n-grams so misspellings still get a vector.",
      "CNN-for-text (Kim 2014): 1-D convolutions + max-pooling detect the most salient learned n-gram, fast and parallel.",
      "LSTM: forget/input/output gates control an additive cell state; the BiLSTM reads both directions so each token sees left and right context.",
      "Transformer: scaled dot-product self-attention, multi-head, wrapped in residual + layer-norm and stacked; BERT pretrains with masked-LM, then a [CLS] head fine-tunes for sentiment. DistilBERT keeps ~97% of accuracy at 40% the size.",
    ],
    math: [
      {
        label: "skip-gram softmax",
        tex: "P(o \\mid c) = \\dfrac{\\exp(u_o^{\\top} v_c)}{\\sum_{w}\\exp(u_w^{\\top} v_c)}",
        note: "Train embeddings so a center word c predicts its context words o; cosine similarity then ≈ semantic similarity.",
      },
      {
        label: "negative sampling",
        tex: "\\log \\sigma(u_o^{\\top} v_c) + \\sum_{j=1}^{k}\\mathbb{E}_{w\\sim P_n}\\big[\\log \\sigma(-u_w^{\\top} v_c)\\big]",
        note: "Avoids the vocabulary-wide softmax: push the true pair up and k random noise pairs down.",
      },
      {
        label: "LSTM gates",
        tex: "\\begin{aligned} f_t &= \\sigma(W_f[h_{t-1}, x_t] + b_f) \\\\ i_t &= \\sigma(W_i[h_{t-1}, x_t] + b_i) \\\\ o_t &= \\sigma(W_o[h_{t-1}, x_t] + b_o) \\end{aligned}",
        note: "Three sigmoid gates in [0,1] decide what to forget, what to write, and what to read out.",
      },
      {
        label: "LSTM cell update",
        tex: "C_t = f_t \\odot C_{t-1} + i_t \\odot \\tanh(W_g[h_{t-1}, x_t] + b_g), \\qquad h_t = o_t \\odot \\tanh(C_t)",
        note: "The additive path through C_t is the 'highway' that lets gradients flow without repeated multiplication.",
      },
      {
        label: "self-attention",
        tex: "\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\Big(\\dfrac{Q K^{\\top}}{\\sqrt{d_k}}\\Big)V",
        note: "Each token's output is a weighted average of all Values; √dₖ keeps dot products from saturating the softmax.",
      },
      {
        label: "BERT masked-LM",
        tex: "\\mathcal{L}_{\\text{MLM}} = -\\sum_{i \\in M}\\log P(x_i \\mid x_{\\setminus M})",
        note: "Predict 15% randomly masked tokens from both-side context → forces bidirectional, contextual embeddings.",
      },
    ],
    workedExample: [
      "Self-attention flips 'good' in 'not good'  (dₖ=1, scalars).",
      "Query of 'good' scores against the keys:",
      "  score(good→not)  = 2.0",
      "  score(good→good) = 1.0",
      "",
      "  softmax([2.0, 1.0]) = [e^2/(e^2+e^1), e^1/(e^2+e^1)]",
      "                      = [7.39/10.10, 2.72/10.10]",
      "                      = [0.731, 0.269]",
      "",
      "  new 'good' = 0.731·V(not) + 0.269·V(good)",
      "  → 'good' literally absorbs 73% of 'not' and turns negative ✓",
    ].join("\n"),
    pros: [
      "Big jump on context, negation scope, and long-range dependencies.",
      "Embeddings remove most manual feature engineering; transfer across tasks.",
      "Transformers parallelize across all tokens — fast to train at scale.",
      "Contextual embeddings solve word-sense ('sick beat' vs 'feeling sick'); DistilBERT is production-friendly.",
    ],
    cons: [
      "Needs more data and a GPU; pre-transformer RNNs are slow (sequential) to train.",
      "Static embeddings (word2vec/GloVe) give one vector per word — sense-blind, and 'good'/'bad' can land near each other.",
      "Far less interpretable than linear models.",
      "Heavier to deploy than a rule-based scorer or an LR classifier.",
    ],
    whenToUse:
      "You have tens of thousands of labels (or a good pretrained checkpoint), GPU budget, and need real handling of context, negation, and word-sense. Fine-tune DistilBERT/twitter-roberta for the best accuracy-to-cost ratio in this tier.",
  },

  // ────────────────────────────────────────────────────────────── 5. LLMs
  {
    id: "llm",
    title: "LLMs (Claude)",
    era: "2022+",
    tagline: "Just ask — zero-shot nuance, structured output, free explanations.",
    intuition:
      "A large language model has already absorbed syntax, world knowledge, and pragmatics from pretraining, so you can skip training data entirely: describe the task in a prompt and it returns a label, aspects, and a rationale in one shot. It is the same Transformer machinery as the deep-learning tier, scaled up and instruction-tuned, so it handles sarcasm, comparatives, and aspect-conflicting reviews that trip every earlier method — at the cost of latency and per-call price.",
    howItWorks: [
      "Prompt the model with the text and an output schema (e.g. ask for JSON: label, confidence, per-aspect polarity, a one-line reason).",
      "Zero- or few-shot: no fine-tuning needed; a handful of in-context examples can sharpen consistency.",
      "Under the hood it is still autoregressive next-token prediction over the same self-attention stack — generality comes from scale + instruction tuning, not a new equation.",
      "Add guardrails (schema validation, temperature control, retries) because free-form generation can drift in format.",
    ],
    math: [
      {
        label: "autoregressive factorization",
        tex: "P(x_1, \\dots, x_T) = \\prod_{t=1}^{T} P(x_t \\mid x_{<t})",
        note: "An LLM is trained to predict the next token given all previous ones — sentiment is just a downstream of that.",
      },
      {
        label: "next-token softmax",
        tex: "P(x_t \\mid x_{<t}) = \\text{softmax}\\!\\big(W\\, h_t / \\tau\\big)",
        note: "Temperature τ controls determinism; τ→0 makes labels reproducible, which you want for classification.",
      },
      {
        label: "in-context learning",
        tex: "\\hat{y} = \\arg\\max_{y} P\\big(y \\mid \\underbrace{(x_1, y_1), \\dots, (x_k, y_k)}_{\\text{few-shot examples}},\\; x\\big)",
        note: "The 'training' happens inside the prompt — no weight updates at all.",
      },
    ],
    workedExample: [
      'Prompt: classify "The plot was predictable, but honestly',
      '         I could not stop watching."',
      "",
      "Claude (zero-shot) →",
      "  {",
      '    "label": "positive",',
      '    "confidence": 0.78,',
      '    "aspects": [',
      '      {"term": "plot", "polarity": "negative"},',
      '      {"term": "watchability", "polarity": "positive"}',
      "    ],",
      '    "reason": "concedes a flaw with \'but\', then the',
      '               contrast clause carries the verdict"',
      "  }",
      "  → catches the 'but' contrast + mixed aspects in one call ✓",
    ].join("\n"),
    pros: [
      "Best accuracy on hard, nuanced text: sarcasm, comparatives, mixed aspects.",
      "Zero labels required; flexible output schema (structured JSON, aspects, ordinal).",
      "Explanations for free — useful for human review and debugging.",
      "One model covers polarity, ABSA, and emotion without separate training runs.",
    ],
    cons: [
      "Latency and per-call cost; rate limits at high volume.",
      "Overkill for simple high-throughput polarity.",
      "Less interpretable internally; needs guardrails for output consistency.",
      "Non-determinism unless temperature is pinned low.",
    ],
    whenToUse:
      "Nuanced or aspect-based sentiment, low-to-moderate volume, no labeled data, or when you need a human-readable rationale. The strongest off-the-shelf option for hard ABSA; pair with a cheap model (VADER/LR) for the easy high-volume traffic.",
  },
];
