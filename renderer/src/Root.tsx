import { Composition } from "remotion";
import { ViolenceHighlights, calculateMetadata } from "./ViolenceHighlights";
import { VerticalHighlights, calculateVerticalMetadata } from "./VerticalHighlights";
import type { ViolenceHighlightsProps } from "./types";

const defaultProps: ViolenceHighlightsProps = {
  source: "source.mp4",
  segments: [],
  showLabels: true,
  fps: 30,
  voiceovers: [],
};

export const RemotionRoot = () => {
  return (
    <>
      <Composition
        id="ViolenceHighlights"
        component={ViolenceHighlights}
        durationInFrames={300}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={defaultProps}
        calculateMetadata={calculateMetadata}
      />
      <Composition
        id="VerticalHighlights"
        component={VerticalHighlights}
        durationInFrames={300}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={defaultProps}
        calculateMetadata={calculateVerticalMetadata}
      />
    </>
  );
};
