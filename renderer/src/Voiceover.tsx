import { Audio } from "@remotion/media";
import { Sequence, useVideoConfig, interpolate, staticFile } from "remotion";

type VoiceoverProps = {
  /** Path to voiceover WAV file (relative to public/) */
  file: string;
  /** Delay in seconds before voiceover starts within the segment */
  delay?: number;
  /** Volume 0-1 */
  volume?: number;
};

/**
 * Plays a voiceover audio clip with a slight delay and fade-in.
 * Designed to layer over gameplay audio without overwhelming it.
 */
export const Voiceover: React.FC<VoiceoverProps> = ({
  file,
  delay = 0.5,
  volume = 0.8,
}) => {
  const { fps } = useVideoConfig();
  const delayFrames = Math.floor(delay * fps);

  return (
    <Sequence from={delayFrames} layout="none">
      <Audio
        src={staticFile(file)}
        volume={(f) =>
          interpolate(f, [0, Math.floor(0.15 * fps)], [0, volume], {
            extrapolateRight: "clamp",
          })
        }
      />
    </Sequence>
  );
};
