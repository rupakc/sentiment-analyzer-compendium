export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  description: string;
  available: boolean;
}
export interface Aspect {
  term: string;
  polarity: string;
}
export interface AnalysisResult {
  model_id: string;
  label: string;
  confidence: number;
  scores: Record<string, number>;
  aspects: Aspect[] | null;
  available: boolean;
  error: string | null;
}
export interface Explanation {
  model_id: string;
  explanation_type: "native" | "ai-narrated";
  summary: string;
  evidence: { label: string; detail?: string; weight?: number }[];
  method: string;
  steps: string[];
  biases: string[];
}
