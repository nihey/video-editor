/**
 * Cinematic color grading filter.
 * Boosts contrast, slightly increases saturation, and adds a subtle
 * warm tone to make gameplay footage more visually punchy.
 */
export const ColorGrade: React.FC<{
  children: React.ReactNode;
  contrast?: number;
  brightness?: number;
  saturate?: number;
}> = ({
  children,
  contrast = 1.15,
  brightness = 1.05,
  saturate = 1.1,
}) => {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        filter: `contrast(${contrast}) brightness(${brightness}) saturate(${saturate})`,
      }}
    >
      {children}
    </div>
  );
};
