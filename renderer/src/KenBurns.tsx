import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

type KenBurnsProps = {
  children: React.ReactNode;
  score: number;
  /** Zoom intensity: how much to zoom based on score (0-1 normalized) */
  intensity?: number;
};

/**
 * Score-driven Ken Burns effect.
 * - High-score segments: punch-in zoom at the start (action moment), then hold
 * - Medium-score: slow creeping zoom in throughout
 * - All segments: subtle drift to keep the frame alive
 */
export const KenBurns: React.FC<KenBurnsProps> = ({
  children,
  score,
  intensity = 1,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Normalize score to 0-1 range for zoom calculations
  // Scores typically range 0.3-0.7, map to 0-1
  const normalizedScore = Math.min(
    1,
    Math.max(0, (score - 0.3) / 0.4),
  );

  // Base zoom: 1.0 (no zoom) to 1.15 (15% zoom) based on score
  const maxZoom = 1 + 0.15 * normalizedScore * intensity;

  // High-score segments get a spring punch-in at the start
  const punchIn =
    normalizedScore > 0.5
      ? spring({
          frame,
          fps,
          config: { damping: 15, stiffness: 80, mass: 0.8 },
          durationInFrames: Math.min(fps * 1.5, durationInFrames),
        })
      : 0;

  // Slow creep zoom throughout the segment
  const creepZoom = interpolate(
    frame,
    [0, durationInFrames],
    [1, 1 + 0.05 * normalizedScore * intensity],
    { extrapolateRight: "clamp" },
  );

  // Combine: punch-in brings to maxZoom quickly, creep adds subtle motion
  const zoom = normalizedScore > 0.5
    ? interpolate(punchIn, [0, 1], [1, maxZoom])
    : creepZoom;

  // Subtle horizontal drift (pan) for visual interest
  // Direction alternates based on score to create variety
  const driftDirection = score > 0.45 ? 1 : -1;
  const driftAmount = 1.5 * normalizedScore * intensity; // max 1.5% drift
  const translateX = interpolate(
    frame,
    [0, durationInFrames],
    [0, driftAmount * driftDirection],
    { extrapolateRight: "clamp" },
  );

  // Slight vertical drift
  const translateY = interpolate(
    frame,
    [0, durationInFrames],
    [0, driftAmount * 0.5 * -driftDirection],
    { extrapolateRight: "clamp" },
  );

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <div
        style={{
          width: "100%",
          height: "100%",
          transform: `scale(${zoom}) translate(${translateX}%, ${translateY}%)`,
          transformOrigin: "center center",
        }}
      >
        {children}
      </div>
    </div>
  );
};
