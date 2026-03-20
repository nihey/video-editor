import { Audio } from "@remotion/media";
import { Sequence, useVideoConfig, interpolate } from "remotion";

/**
 * Sound effect configuration for transitions and impacts.
 * Uses Remotion's built-in SFX library + custom bundled sounds.
 */

// Remotion built-in SFX (always available, no download needed)
const REMOTION_SFX = {
  whoosh: "https://remotion.media/whoosh.wav",
  whip: "https://remotion.media/whip.wav",
  vineBoom: "https://remotion.media/vine-boom.wav",
};

type TransitionSFXProps = {
  /** Frame offset from segment start to play the SFX */
  atFrame?: number;
  /** Volume 0-1 */
  volume?: number;
  /** Which sound to play */
  type: "whoosh" | "whip" | "impact";
};

/**
 * Plays a transition SFX at a specific frame within a segment.
 * Wrap in a Sequence to position it at the right time.
 */
export const TransitionSFX: React.FC<TransitionSFXProps> = ({
  atFrame = 0,
  volume = 0.3,
  type,
}) => {
  const src =
    type === "whoosh"
      ? REMOTION_SFX.whoosh
      : type === "whip"
        ? REMOTION_SFX.whip
        : REMOTION_SFX.vineBoom;

  return (
    <Sequence from={atFrame} layout="none">
      <Audio src={src} volume={volume} />
    </Sequence>
  );
};

type SegmentSFXProps = {
  score: number;
  durationInFrames: number;
};

/**
 * Automatically adds sound effects to a segment based on its violence score.
 * - High scores (>0.5): impact boom at start + whoosh at end
 * - Medium scores (0.35-0.5): whoosh at transitions
 * - Low scores: no SFX (let the gameplay audio breathe)
 */
export const SegmentSFX: React.FC<SegmentSFXProps> = ({
  score,
  durationInFrames,
}) => {
  const { fps } = useVideoConfig();
  const normalizedScore = Math.min(1, Math.max(0, (score - 0.3) / 0.4));

  // Volume scales with score — louder for more intense moments
  const baseVolume = interpolate(normalizedScore, [0, 1], [0.1, 0.35], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  if (normalizedScore < 0.15) {
    // Low score: no SFX
    return null;
  }

  return (
    <>
      {/* Entry whoosh for all qualifying segments */}
      <TransitionSFX atFrame={0} type="whoosh" volume={baseVolume * 0.7} />

      {/* Impact boom for high-intensity segments */}
      {normalizedScore > 0.5 && (
        <TransitionSFX
          atFrame={Math.floor(0.1 * fps)}
          type="impact"
          volume={baseVolume}
        />
      )}

      {/* Exit whip sound for segments longer than 3s */}
      {durationInFrames > 3 * fps && (
        <TransitionSFX
          atFrame={Math.max(0, durationInFrames - Math.floor(0.5 * fps))}
          type="whip"
          volume={baseVolume * 0.5}
        />
      )}
    </>
  );
};
