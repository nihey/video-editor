import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

/**
 * Dynamic action text overlay that appears at the start of high-scoring segments.
 * Shows intensity-based callouts like "INTENSE!", "CHAOS!", etc.
 */
export const ActionOverlay: React.FC<{
  score: number;
  label: string;
}> = ({ score, label }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const normalizedScore = Math.min(1, Math.max(0, (score - 0.3) / 0.4));

  // Only show for high-scoring segments
  if (normalizedScore < 0.4) return null;

  // Pick text based on score intensity
  const callouts: Record<string, string[]> = {
    high: ["CHAOS!", "BRUTAL!", "CARNAGE!", "NO MERCY!", "SAVAGE!"],
    mid: ["INTENSE!", "ACTION!", "FIGHT!", "SHOWDOWN!"],
  };

  const tier = normalizedScore > 0.7 ? "high" : "mid";
  const texts = callouts[tier];
  const idx = label.charCodeAt(0) % texts.length;
  const text = texts[idx];

  // Spring scale animation — pops in then settles
  const scale = spring({
    frame,
    fps,
    config: { damping: 8, stiffness: 150, mass: 0.5 },
    durationInFrames: fps * 0.8,
  });

  // Fade out after 1.5s
  const opacity = interpolate(
    frame,
    [0, fps * 0.3, fps * 1.2, fps * 1.8],
    [0, 1, 1, 0],
    { extrapolateRight: "clamp" },
  );

  if (opacity <= 0) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: "15%",
        left: 0,
        width: "100%",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        opacity,
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          fontSize: normalizedScore > 0.7 ? 72 : 56,
          fontWeight: 900,
          fontFamily: "Impact, Arial Black, sans-serif",
          color: "#fff",
          textShadow:
            "0 0 20px rgba(230, 57, 70, 0.8), " +
            "0 0 40px rgba(230, 57, 70, 0.4), " +
            "2px 2px 0 #000, -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000",
          transform: `scale(${scale})`,
          letterSpacing: 4,
        }}
      >
        {text}
      </div>
    </div>
  );
};

/**
 * Score indicator — shows a small intensity meter in the corner.
 */
export const ScoreIndicator: React.FC<{
  score: number;
}> = ({ score }) => {
  const normalizedScore = Math.min(1, Math.max(0, (score - 0.3) / 0.4));
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [0, fps * 0.5], [0, 0.8], {
    extrapolateRight: "clamp",
  });

  const barWidth = normalizedScore * 100;
  const color =
    normalizedScore > 0.7
      ? "#e63946"
      : normalizedScore > 0.4
        ? "#f4a261"
        : "#2a9d8f";

  return (
    <div
      style={{
        position: "absolute",
        bottom: 16,
        right: 16,
        opacity,
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <div
        style={{
          fontSize: 14,
          fontFamily: "monospace",
          fontWeight: "bold",
          color: "#fff",
          textShadow: "1px 1px 2px #000",
        }}
      >
        {Math.round(normalizedScore * 100)}%
      </div>
      <div
        style={{
          width: 60,
          height: 6,
          backgroundColor: "rgba(255,255,255,0.2)",
          borderRadius: 3,
        }}
      >
        <div
          style={{
            width: `${barWidth}%`,
            height: "100%",
            backgroundColor: color,
            borderRadius: 3,
          }}
        />
      </div>
    </div>
  );
};
