import type { CanvasCoordinates, RadarDot } from "../../types.ts";
import { MAX_DOTS } from "./config.ts";

// Canvas and context (to be initialized)
let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let width: number,
  height: number,
  centerX: number,
  centerY: number,
  maxRadius: number;

// Offscreen canvas for static radar base (optimization)
let baseCanvas: HTMLCanvasElement;
let baseCtx: CanvasRenderingContext2D;

// Initialize canvas dimensions and context
export function initCanvas(): void {
  canvas = document.getElementById("radar") as HTMLCanvasElement;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Failed to get 2D context");
  ctx = context;

  // Calculate size based on viewport - use the smaller dimension to keep it circular
  const size = Math.min(window.innerWidth, window.innerHeight) * 0.95; // 95% of viewport
  canvas.width = size;
  canvas.height = size;

  width = canvas.width;
  height = canvas.height;
  centerX = width / 2;
  centerY = height / 2;
  maxRadius = Math.min(width, height) / 2 - 20;

  // Create offscreen canvas for static base
  baseCanvas = document.createElement("canvas");
  baseCanvas.width = width;
  baseCanvas.height = height;
  const baseContext = baseCanvas.getContext("2d");
  if (!baseContext) throw new Error("Failed to get 2D context for base canvas");
  baseCtx = baseContext;

  // Draw the static base once to the offscreen canvas
  drawStaticBase();
}

// Draw the static radar base once to offscreen canvas
function drawStaticBase(): void {
  // Draw concentric circles (distance rings) with distance labels
  baseCtx.strokeStyle = "rgba(0, 255, 255, 0.15)";
  baseCtx.lineWidth = 1;
  baseCtx.font = "20px monospace";
  baseCtx.fillStyle = "rgba(0, 255, 255, 0.5)";
  baseCtx.textAlign = "center";
  baseCtx.textBaseline = "middle";

  for (let i = 1; i <= 8; i++) {
    const radius = (maxRadius / 8) * i;
    // Draw circle
    baseCtx.beginPath();
    baseCtx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    baseCtx.stroke();

    // Calculate distance in meters (10m increments up to 80m)
    const distance = 10 * i;

    // Draw distance label at the top of each circle
    baseCtx.fillText(`${distance}m`, centerX, centerY - radius - 8);
  }

  // Draw crosshairs (quadrant dividers)
  baseCtx.strokeStyle = "rgba(0, 255, 255, 0.25)";
  baseCtx.lineWidth = 1.5;

  // Vertical line
  baseCtx.beginPath();
  baseCtx.moveTo(centerX, centerY - maxRadius);
  baseCtx.lineTo(centerX, centerY + maxRadius);
  baseCtx.stroke();

  // Horizontal line
  baseCtx.beginPath();
  baseCtx.moveTo(centerX - maxRadius, centerY);
  baseCtx.lineTo(centerX + maxRadius, centerY);
  baseCtx.stroke();

  // Draw center point
  baseCtx.fillStyle = "rgba(0, 255, 255, 0.6)";
  baseCtx.beginPath();
  baseCtx.arc(centerX, centerY, 4, 0, Math.PI * 2);
  baseCtx.fill();

  // Draw outer border
  baseCtx.strokeStyle = "rgba(0, 255, 255, 0.3)";
  baseCtx.lineWidth = 2;
  baseCtx.beginPath();
  baseCtx.arc(centerX, centerY, maxRadius, 0, Math.PI * 2);
  baseCtx.stroke();
}

// Composite the static base onto main canvas (called each frame)
export function drawRadarBase(): void {
  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  // Draw the pre-rendered base from offscreen canvas
  ctx.drawImage(baseCanvas, 0, 0);
}

// Convert polar coordinates (from CSV) to canvas coordinates
export function cartesianToCanvas(
  x: number,
  y: number,
  range: number
): CanvasCoordinates {
  // Calculate polar angle from x,y components using atan2
  const angle: number = Math.atan2(y, x);

  // Scale factor: range is in meters, we need to scale to fit radar
  // Max range is 80 meters to fit in the radar circle
  const scale: number = maxRadius / 80;

  // Convert from polar (angle, range) to canvas Cartesian coordinates
  // Canvas: (0,0) is top-left, positive x is right, positive y is down
  // Radar: angle 0 is to the right (east), angle increases counter-clockwise
  const scaledRange: number = range * scale;
  const canvasX: number = centerX + scaledRange * Math.cos(angle);
  const canvasY: number = centerY - scaledRange * Math.sin(angle); // Invert y because canvas y is down

  return { x: canvasX, y: canvasY };
}

// Generate color based on ID number (warm colors: red, orange, yellow)
function getColorForId(id: number): { r: number; g: number; b: number } {
  // Use ID to generate a hue in the warm color range (0-60 degrees)
  // Red = 0°, Orange = 30°, Yellow = 60°
  const hue = (id * 137.5) % 60; // Golden angle distribution for better spread

  // Convert HSL to RGB (fixed saturation and lightness for vibrant colors)
  const s = 0.9; // 90% saturation
  const l = 0.6; // 60% lightness

  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0, g = 0, b = 0;

  if (hue < 60) {
    r = c;
    g = x;
    b = 0;
  }

  return {
    r: Math.round((r + m) * 255),
    g: Math.round((g + m) * 255),
    b: Math.round((b + m) * 255)
  };
}

// Draw tooltip for a single dot
function drawDotTooltip(dot: RadarDot, opacity: number, color: { r: number; g: number; b: number }): void {
  const tooltipOffset: number = 15;
  const tooltipX: number = dot.canvasX + tooltipOffset;
  const tooltipY: number = dot.canvasY - tooltipOffset;
  const padding: number = 10;
  const lineHeight: number = 16;

  // Prepare text - show track_id, class, x, y, range, velocity
  const texts: string[] = [
    `ID: ${dot.track_id || "?"}`,
    `Class: ${dot.class || "?"}`,
    `X: ${dot.x !== undefined ? dot.x.toFixed(2) : "?"}`,
    `Y: ${dot.y !== undefined ? dot.y.toFixed(2) : "?"}`,
    `Range: ${dot.range ? `${dot.range.toFixed(2)}m` : "?"}`,
    `V: ${dot.velocity ? `${dot.velocity.toFixed(1)}m/s` : "?"}`,
  ];

  // Set font for measurement
  ctx.font = "14px monospace";

  // Calculate tooltip dimensions
  const maxWidth: number = Math.max(
    ...texts.map((t) => ctx.measureText(t).width)
  );
  const tooltipWidth: number = maxWidth + padding * 2;
  const tooltipHeight: number = texts.length * lineHeight + padding * 2;

  // Draw tooltip background
  ctx.fillStyle = `rgba(10, 14, 39, ${0.9 * opacity})`;
  ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${0.5 * opacity})`;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 3);
  ctx.fill();
  ctx.stroke();

  // Draw text lines
  ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
  ctx.font = "14px monospace";
  texts.forEach((text, i) => {
    ctx.fillText(
      text,
      tooltipX + padding,
      tooltipY + padding + lineHeight * (i + 1) - 2
    );
  });
}

// Draw all radar dots
export function drawRadarDots(radarDots: RadarDot[]): void {
  radarDots.forEach((dot, index) => {
    // Calculate opacity based on age (newer dots are brighter)
    const age: number = radarDots.length - index;
    const opacity: number = Math.max(0.1, 1 - age / MAX_DOTS);

    // Get color based on track_id
    const color = getColorForId(dot.track_id || 0);
    const colorString = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;

    // Draw dot
    ctx.fillStyle = colorString;
    ctx.beginPath();
    ctx.arc(dot.canvasX, dot.canvasY, 5, 0, Math.PI * 2);
    ctx.fill();

    // Add glow effect to newest dots
    if (index === radarDots.length - 1) {
      ctx.shadowBlur = 15;
      ctx.shadowColor = `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`;
      ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 1)`;
      ctx.beginPath();
      ctx.arc(dot.canvasX, dot.canvasY, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Draw tooltip for each dot
    drawDotTooltip(dot, opacity, color);
  });
}
