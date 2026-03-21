import { AbsoluteFill, useVideoConfig, staticFile, useCurrentFrame, interpolate, spring } from "remotion";
import type { CalculateMetadataFunction } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { Video } from "@remotion/media";
import { SegmentLabel } from "./SegmentLabel";
import { SegmentSFX } from "./SoundEffects";
import { Voiceover } from "./Voiceover";
import { ColorGrade } from "./ColorGrade";
import { ActionOverlay, ScoreIndicator } from "./ActionOverlay";
import { ProgressBar } from "./ProgressBar";
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
  /** Focal point X (0=left, 0.5=center, 1=right) from motion analysis */
  focalX?: number;
}> = ({ children, score, focalX = 0.5 }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const normalizedScore = Math.min(1, Math.max(0, (score - 0.3) / 0.4));

  // Zoom
  const maxZoom = 1 + 0.25 * normalizedScore;
  const punchIn =
    normalizedScore > 0.5
      ? spring({
          frame,
          fps,
          config: { damping: 12, stiffness: 100, mass: 0.6 },
          durationInFrames: Math.min(fps * 1, durationInFrames),
        })
      : 0;

  const creepZoom = interpolate(
    frame,
    [0, durationInFrames],
    [1, 1 + 0.1 * normalizedScore],
    { extrapolateRight: "clamp" },
  );

  const zoom =
    normalizedScore > 0.5
      ? interpolate(punchIn, [0, 1], [1, maxZoom])
      : creepZoom;

  // Smart horizontal offset based on focal point detection
  // focalX=0.3 means action is left-of-center, pan left to show it
  // Convert focal_x (0-1) to translate offset (% of frame)
  // In vertical crop we see ~56% of the horizontal frame (9/16),
  // so we have ~22% on each side to shift
  const focalOffset = (focalX - 0.5) * -18; // shift opposite to focal point

  // Drift adds motion on top of the focal offset
  const driftDirection = score > 0.45 ? 1 : -1;
  const driftAmount = 3 * normalizedScore;
  const translateX = focalOffset + interpolate(
    frame,
    [0, durationInFrames],
    [-driftAmount * 0.3 * driftDirection, driftAmount * 0.3 * driftDirection],
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
          <ColorGrade>
            <VerticalKenBurns score={seg.score} focalX={seg.focal_x}>
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
          </ColorGrade>
          <ActionOverlay score={seg.score} label={seg.label} />
          <ScoreIndicator score={seg.score} />
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
      <ProgressBar />
    </AbsoluteFill>
  );
};
