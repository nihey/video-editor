import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";

/**
 * Thin progress bar at the bottom of the video.
 * Shows how far through the compilation the viewer is — a retention signal.
 */
export const ProgressBar: React.FC<{
  color?: string;
  height?: number;
}> = ({ color = "#e63946", height = 4 }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const progress = interpolate(frame, [0, durationInFrames], [0, 100], {
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        bottom: 0,
        left: 0,
        width: "100%",
        height,
        backgroundColor: "rgba(255, 255, 255, 0.15)",
      }}
    >
      <div
        style={{
          width: `${progress}%`,
          height: "100%",
          backgroundColor: color,
          transition: "none",
        }}
      />
    </div>
  );
};
