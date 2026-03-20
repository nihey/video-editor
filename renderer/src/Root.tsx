import { Composition } from "remotion";
import { ViolenceHighlights, calculateMetadata } from "./ViolenceHighlights";
import type { ViolenceHighlightsProps } from "./types";

export const RemotionRoot = () => {
  return (
    <Composition
      id="ViolenceHighlights"
      component={ViolenceHighlights}
      durationInFrames={300}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={
        {
          source: "source.mp4",
          segments: [],
          showLabels: true,
          fps: 30,
        } satisfies ViolenceHighlightsProps
      }
      calculateMetadata={calculateMetadata}
    />
  );
};
