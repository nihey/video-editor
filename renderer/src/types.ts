export type Segment = {
  label: string;
  start: number;
  end: number;
  duration: number;
  score: number;
  /** Focal X position for vertical crop (0.0=left, 0.5=center, 1.0=right) */
  focal_x?: number;
};

export type VoiceoverEntry = {
  label: string;
  file: string;
};

export type ViolenceHighlightsProps = {
  source: string;
  segments: Segment[];
  showLabels: boolean;
  fps: number;
  voiceovers?: VoiceoverEntry[];
};
