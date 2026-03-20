import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";

export const SegmentLabel: React.FC<{ label: string }> = ({ label }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Fade in over 0.3s at the start of each segment
  const opacity = interpolate(frame, [0, 0.3 * fps], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        top: 20,
        left: 20,
        backgroundColor: "rgba(0, 0, 0, 0.6)",
        color: "#fff",
        fontSize: 36,
        fontWeight: "bold",
        fontFamily: "monospace",
        padding: "6px 16px",
        borderRadius: 6,
        opacity,
        letterSpacing: 2,
      }}
    >
      {label}
    </div>
  );
};
