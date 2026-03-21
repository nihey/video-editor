import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

type KenBurnsProps = {
  children: React.ReactNode;
  score: number;
  intensity?: number;
};

/**
 * Score-driven Ken Burns effect.
 * - High-score segments: fast spring punch-in zoom, then slow pull-back
 * - Medium-score: slow creeping zoom in throughout
 * - All segments: visible drift pan to keep the frame alive
 */
export const KenBurns: React.FC<KenBurnsProps> = ({
  children,
  score,
  intensity = 1,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const normalizedScore = Math.min(1, Math.max(0, (score - 0.3) / 0.4));

  // High-score: punch in to 1.35x then slowly pull back to 1.15x
  // Medium: creep from 1.0 to 1.15x
  // Low: creep from 1.0 to 1.08x
  const punchTarget = 1 + 0.35 * normalizedScore * intensity;
  const holdTarget = 1 + 0.15 * normalizedScore * intensity;
  const creepTarget = 1 + 0.12 * Math.max(0.3, normalizedScore) * intensity;

  let zoom: number;

  if (normalizedScore > 0.5) {
    // Punch in quickly, then slowly ease back
    const punchPhase = spring({
      frame,
      fps,
      config: { damping: 12, stiffness: 100, mass: 0.6 },
      durationInFrames: Math.min(fps * 1, durationInFrames),
    });

    const pullBack = interpolate(
      frame,
      [fps * 0.8, durationInFrames],
      [0, 1],
      { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
    );

    const punchedZoom = interpolate(punchPhase, [0, 1], [1, punchTarget]);
    zoom = punchedZoom - pullBack * (punchTarget - holdTarget);
  } else {
    // Smooth creep zoom
    zoom = interpolate(
      frame,
      [0, durationInFrames],
      [1, creepTarget],
      { extrapolateRight: "clamp" },
    );
  }

  // Pan drift — visible movement across the frame
  const driftDirection = score > 0.45 ? 1 : -1;
  const driftAmount = 3 * Math.max(0.3, normalizedScore) * intensity;
  const translateX = interpolate(
    frame,
    [0, durationInFrames],
    [-driftAmount * 0.5 * driftDirection, driftAmount * 0.5 * driftDirection],
    { extrapolateRight: "clamp" },
  );

  const translateY = interpolate(
    frame,
    [0, durationInFrames],
    [driftAmount * 0.3 * driftDirection, -driftAmount * 0.3 * driftDirection],
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
