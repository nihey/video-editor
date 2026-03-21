import { AbsoluteFill, useVideoConfig, staticFile, useCurrentFrame, interpolate, spring } from "remotion";
import type { CalculateMetadataFunction } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { Video } from "@remotion/media";
import { SegmentLabel } from "./SegmentLabel";
import { SegmentSFX } from "./SoundEffects";
import { Voiceover } from "./Voiceover";
import type { ViolenceHighlightsProps, Segment } from "./types";

const TRANSITION_FRAMES = 10;

export const calculateVerticalMetadata: CalculateMetadataFunction<
  ViolenceHighlightsProps
> = async ({ props }) => {
  const fps = props.fps || 30;
  const totalSegmentFrames = props.segments.reduce(
    (sum: number, seg: Segment) => sum + Math.ceil(seg.duration * fps),
    0,
  );
  const transitionCount = Math.max(0, props.segments.length - 1);
  const totalFrames = totalSegmentFrames - transitionCount * TRANSITION_FRAMES;

  return {
    durationInFrames: Math.max(1, totalFrames),
    fps,
    width: 1080,
    height: 1920,
  };
};

/**
 * Ken Burns effect tuned for vertical crop.
 * More aggressive horizontal panning to sweep across the wide source footage,
 * since we're only showing ~60% of the horizontal frame.
 */
const VerticalKenBurns: React.FC<{
  children: React.ReactNode;
  score: number;
}> = ({ children, score }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const normalizedScore = Math.min(1, Math.max(0, (score - 0.3) / 0.4));

  // Zoom: vertical needs more zoom to fill the frame interestingly
  const maxZoom = 1 + 0.2 * normalizedScore;

  const punchIn =
    normalizedScore > 0.5
      ? spring({
          frame,
          fps,
          config: { damping: 15, stiffness: 80, mass: 0.8 },
          durationInFrames: Math.min(fps * 1.5, durationInFrames),
        })
      : 0;

  const creepZoom = interpolate(
    frame,
    [0, durationInFrames],
    [1, 1 + 0.08 * normalizedScore],
    { extrapolateRight: "clamp" },
  );

  const zoom =
    normalizedScore > 0.5
      ? interpolate(punchIn, [0, 1], [1, maxZoom])
      : creepZoom;

  // More aggressive horizontal pan for vertical — sweep across the wide source
  // This makes the vertical version feel alive instead of a static center crop
  const driftDirection = score > 0.45 ? 1 : -1;
  const driftAmount = 4 * normalizedScore; // up to 4% horizontal sweep
  const translateX = interpolate(
    frame,
    [0, durationInFrames],
    [-driftAmount * 0.5 * driftDirection, driftAmount * 0.5 * driftDirection],
    { extrapolateRight: "clamp" },
  );

  const translateY = interpolate(
    frame,
    [0, durationInFrames],
    [0, 1 * normalizedScore * -driftDirection],
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

export const VerticalHighlights: React.FC<ViolenceHighlightsProps> = ({
  source,
  segments,
  showLabels,
  voiceovers,
}) => {
  const { fps } = useVideoConfig();
  const voiceoverMap = new Map(
    (voiceovers || []).map((vo) => [vo.label, vo.file]),
  );

  const children: React.ReactNode[] = [];

  segments.forEach((seg, i) => {
    const durationInFrames = Math.ceil(seg.duration * fps);
    const trimBefore = Math.floor(seg.start * fps);
    const trimAfter = Math.floor(seg.end * fps);

    if (i > 0) {
      children.push(
        <TransitionSeries.Transition
          key={`transition-${seg.label}`}
          presentation={fade()}
          timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
        />,
      );
    }

    children.push(
      <TransitionSeries.Sequence
        key={seg.label}
        durationInFrames={durationInFrames}
      >
        <AbsoluteFill>
          <VerticalKenBurns score={seg.score}>
            {/* Scale video to fill vertical frame width — this crops top/bottom */}
            <Video
              src={staticFile(source)}
              trimBefore={trimBefore}
              trimAfter={trimAfter}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
              }}
            />
          </VerticalKenBurns>
          {showLabels && <SegmentLabel label={seg.label} />}
          <SegmentSFX score={seg.score} durationInFrames={durationInFrames} />
          {voiceoverMap.has(seg.label) && (
            <Voiceover file={voiceoverMap.get(seg.label)!} />
          )}
        </AbsoluteFill>
      </TransitionSeries.Sequence>,
    );
  });

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      <TransitionSeries>{children}</TransitionSeries>
    </AbsoluteFill>
  );
};
