import { AbsoluteFill, Series, useVideoConfig, staticFile } from "remotion";
import type { CalculateMetadataFunction } from "remotion";
import { Video } from "@remotion/media";
import { SegmentLabel } from "./SegmentLabel";
import type { ViolenceHighlightsProps, Segment } from "./types";

export const calculateMetadata: CalculateMetadataFunction<
  ViolenceHighlightsProps
> = async ({ props }) => {
  const totalDuration = props.segments.reduce(
    (sum: number, seg: Segment) => sum + seg.duration,
    0,
  );
  const fps = props.fps || 30;

  return {
    durationInFrames: Math.ceil(totalDuration * fps),
    fps,
  };
};

export const ViolenceHighlights: React.FC<ViolenceHighlightsProps> = ({
  source,
  segments,
  showLabels,
}) => {
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      <Series>
        {segments.map((seg) => {
          const durationInFrames = Math.ceil(seg.duration * fps);
          const trimBefore = Math.floor(seg.start * fps);
          const trimAfter = Math.floor(seg.end * fps);

          return (
            <Series.Sequence key={seg.label} durationInFrames={durationInFrames}>
              <AbsoluteFill>
                <Video
                  src={staticFile(source)}
                  trimBefore={trimBefore}
                  trimAfter={trimAfter}
                />
                {showLabels && <SegmentLabel label={seg.label} />}
              </AbsoluteFill>
            </Series.Sequence>
          );
        })}
      </Series>
    </AbsoluteFill>
  );
};
