export type Segment = {
  label: string;
  start: number;
  end: number;
  duration: number;
  score: number;
};

export type ViolenceHighlightsProps = {
  source: string;
  segments: Segment[];
  showLabels: boolean;
  fps: number;
};
