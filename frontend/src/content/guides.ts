export interface Guide {
  id: string;
  title: string;
  era: string;
  body: string;
}

export const GUIDES: Guide[] = [
  {
    id: "lexicon",
    title: "Lexicon / Rule-based (VADER)",
    era: "≈1997–2010",
    body: "Dictionary of word→polarity scores, summed with rules for negation, intensifiers, and punctuation. Worked example: 'not very good' → +1.9 boosted to +2.193, negated ×-0.74 = -1.623, normalized to -0.39. Zero training, fully interpretable.",
  },
  {
    id: "classical",
    title: "Classical ML (Logistic Regression)",
    era: "2002+",
    body: "TF-IDF n-grams → a discriminative classifier. Logistic Regression models P(class|text) directly, avoiding Naïve Bayes's independence assumption. Its gradient is (prediction − truth)·features — the single-neuron base case of every neural network that followed.",
  },
  {
    id: "sequence",
    title: "Sequence models (HMM & CRF)",
    era: "2004–2015",
    body: "Label tokens, not whole documents — used for aspect extraction. HMMs are generative (Viterbi decoding). CRFs are discriminative and globally normalized, so they admit arbitrary overlapping features and avoid the label-bias problem — exactly why they beat HMMs on tagging.",
  },
  {
    id: "deep",
    title: "Deep learning (CNN/LSTM → DistilBERT)",
    era: "2013+",
    body: "Learned embeddings replace hand-built features. LSTMs store a 'negation pending' signal in their cell state. Transformers use self-attention: in 'not good', the word 'good' absorbs most of 'not' via softmax-weighted attention. DistilBERT is a compressed BERT — ~97% of the accuracy at 40% the size.",
  },
  {
    id: "llm",
    title: "LLMs (Claude)",
    era: "2022+",
    body: "Zero-shot via prompting: no training data, handles sarcasm and aspect-based sentiment, and returns structured output with a natural-language rationale. Best on nuanced text; the cost is an API call per analysis.",
  },
];
