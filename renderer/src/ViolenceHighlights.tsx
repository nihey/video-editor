import { AbsoluteFill, useVideoConfig, staticFile } from "remotion";
import type { CalculateMetadataFunction } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { Video } from "@remotion/media";
import { SegmentLabel } from "./SegmentLabel";
import { KenBurns } from "./KenBurns";
import { SegmentSFX } from "./SoundEffects";
import type { ViolenceHighlightsProps, Segment } from "./types";

const TRANSITION_FRAMES = 10; // ~0.33s fade between segments

export const calculateMetadata: CalculateMetadataFunction<
  ViolenceHighlightsProps
> = async ({ props }) => {
  const fps = props.fps || 30;
  const totalSegmentFrames = props.segments.reduce(
    (sum: number, seg: Segment) => sum + Math.ceil(seg.duration * fps),
    0,
  );
  // Transitions overlap, subtract their duration
  const transitionCount = Math.max(0, props.segments.length - 1);
  const totalFrames = totalSegmentFrames - transitionCount * TRANSITION_FRAMES;

  return {
    durationInFrames: Math.max(1, totalFrames),
    fps,
  };
};

export const ViolenceHighlights: React.FC<ViolenceHighlightsProps> = ({
  source,
  segments,
  showLabels,
}) => {
  const { fps } = useVideoConfig();

  const children: React.ReactNode[] = [];

  segments.forEach((seg, i) => {
    const durationInFrames = Math.ceil(seg.duration * fps);
    const trimBefore = Math.floor(seg.start * fps);
    const trimAfter = Math.floor(seg.end * fps);

    // Add transition between segments (not before the first)
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
          <KenBurns score={seg.score}>
            <Video
              src={staticFile(source)}
              trimBefore={trimBefore}
              trimAfter={trimAfter}
              style={{ width: "100%", height: "100%" }}
            />
          </KenBurns>
          {showLabels && <SegmentLabel label={seg.label} />}
          <SegmentSFX score={seg.score} durationInFrames={durationInFrames} />
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
