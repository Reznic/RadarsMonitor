import type {
	CanvasCoordinates,
	RadarDot,
	RadarsStatusResponse,
} from "../../../types.ts";
import { MAX_DOTS } from "../config.ts";
import { getTooltipConfig } from "../debugConfig.ts";

// Canvas and context (to be initialized)
let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let width: number,
	height: number,
	centerX: number,
	centerY: number,
	maxRadius: number;

const VEHICLE_IMAGE_SRC = "assets/namer.png";
let vehicleImage: HTMLImageElement | null = null;
let vehicleImageLoaded = false;

// Offscreen canvas for static radar base (optimization)
let baseCanvas: HTMLCanvasElement;
let baseCtx: CanvasRenderingContext2D;

// Sweep line state for health check indicator
let sweepAngle: number = 0; // Current angle in radians
let lastSweepUpdate: number = Date.now();

// Initialize canvas dimensions and context
export function initCanvas(): void {
	canvas = document.getElementById("radar") as HTMLCanvasElement;
	const context = canvas.getContext("2d");
	if (!context) throw new Error("Failed to get 2D context");
	ctx = context;

	ensureVehicleImageRequested();

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

function ensureVehicleImageRequested(): void {
	if (vehicleImage || vehicleImageLoaded) {
		return;
	}

	vehicleImage = new Image();
	vehicleImage.src = VEHICLE_IMAGE_SRC;
	vehicleImage.onload = () => {
		vehicleImageLoaded = true;
		if (baseCtx) {
			drawStaticBase();
		}
	};
	vehicleImage.onerror = (error) => {
		console.error("Failed to load vehicle image", error);
	};
}

// Draw the static radar base once to offscreen canvas
function drawStaticBase(): void {
	baseCtx.clearRect(0, 0, width, height);

	// Draw concentric circles (distance rings) with distance labels
	baseCtx.strokeStyle = "rgba(0, 255, 255, 0.15)";
	baseCtx.lineWidth = 1;
	baseCtx.font = "20px monospace";
	baseCtx.fillStyle = "rgba(0, 255, 255, 0.5)";
	baseCtx.textAlign = "center";
	baseCtx.textBaseline = "middle";

	for (let i = 1; i <= 5; i++) {
		const radius = (maxRadius / 5) * i;
		// Draw circle
		baseCtx.beginPath();
		baseCtx.arc(centerX, centerY, radius, 0, Math.PI * 2);
		baseCtx.stroke();

		// Calculate distance in meters (10m increments up to 50m)
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

	// Draw outer border
	baseCtx.strokeStyle = "rgba(0, 255, 255, 0.3)";
	baseCtx.lineWidth = 2;
	baseCtx.beginPath();
	baseCtx.arc(centerX, centerY, maxRadius, 0, Math.PI * 2);
	baseCtx.stroke();
}

export function drawVehicleOverlay(): void {
	if (!vehicleImage || !vehicleImageLoaded) {
		return;
	}

	const aspectRatio =
		vehicleImage.naturalWidth && vehicleImage.naturalHeight
			? vehicleImage.naturalWidth / vehicleImage.naturalHeight
			: 0.667; // Fallback to original ratio if metadata is missing

	const vehicleHeight = maxRadius * 0.2;
	const vehicleWidth = vehicleHeight * aspectRatio;
	const imageX = centerX - vehicleWidth / 2;
	const imageY = centerY - vehicleHeight / 2;

	ctx.save();
	ctx.globalAlpha = 0.9;
	ctx.imageSmoothingEnabled = true;
	ctx.drawImage(vehicleImage, imageX, imageY, vehicleWidth, vehicleHeight);
	ctx.restore();

	drawVehicleCornerMarkers(ctx, imageX, imageY, vehicleWidth, vehicleHeight);
}

function drawVehicleCornerMarkers(
	targetCtx: CanvasRenderingContext2D,
	imageX: number,
	imageY: number,
	imageWidth: number,
	imageHeight: number,
): void {
	const padding = 14; //Math.max(2, Math.min(imageWidth, imageHeight) * 0.0008);
	const cornerLength = Math.max(5, Math.min(imageWidth, imageHeight) * 0.018);
	const dotRadius = 4;

	const corners = [
		{ label: "NW", x: imageX + padding, y: imageY + padding, h: 1, v: 1 },
		{
			label: "NE",
			x: imageX + imageWidth - padding,
			y: imageY + padding,
			h: -1,
			v: 1,
		},
		{
			label: "SW",
			x: imageX + padding,
			y: imageY + imageHeight - padding,
			h: 1,
			v: -1,
		},
		{
			label: "SE",
			x: imageX + imageWidth - padding,
			y: imageY + imageHeight - padding,
			h: -1,
			v: -1,
		},
	];

	targetCtx.save();
	targetCtx.lineWidth = 2;
	targetCtx.strokeStyle = "rgba(0, 255, 255, 0.55)";
	targetCtx.fillStyle = "rgba(0, 255, 255, 0.9)";
	targetCtx.font = "11px monospace";
	targetCtx.textAlign = "center";
	targetCtx.textBaseline = "middle";

	corners.forEach(({ x, y, h, v }) => {
		targetCtx.beginPath();
		targetCtx.moveTo(x, y + v * cornerLength);
		targetCtx.lineTo(x, y);
		targetCtx.lineTo(x + h * cornerLength, y);
		targetCtx.stroke();

		const dotX = x;
		const dotY = y;
		targetCtx.beginPath();
		targetCtx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
		targetCtx.fill();
	});

	targetCtx.restore();
}

// Composite the static base onto main canvas (called each frame)
export function drawRadarBase(): void {
	// Clear canvas
	ctx.clearRect(0, 0, width, height);

	// Draw the pre-rendered base from offscreen canvas
	ctx.drawImage(baseCanvas, 0, 0);
}

// Convert polar coordinates to canvas coordinates
export function cartesianToCanvas(x: number, y: number): CanvasCoordinates {
	// Scale component values (meters) directly into canvas space
	const scale: number = maxRadius / 50; // Max visualized range: 50m
	const canvasX: number = centerX + x * scale;
	const canvasY: number = centerY - y * scale; // Invert to keep positive Y pointing up

	return { x: canvasX, y: canvasY };
}

function radarAngleToCanvas(angleDegrees: number): number {
	const normalized = (((angleDegrees - 90) % 360) + 360) % 360;
	return (normalized * Math.PI) / 180;
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

	let r = 0,
		g = 0,
		b = 0;

	if (hue < 60) {
		r = c;
		g = x;
		b = 0;
	}

	return {
		r: Math.round((r + m) * 255),
		g: Math.round((g + m) * 255),
		b: Math.round((b + m) * 255),
	};
}

// Draw tooltip for a single dot
function drawDotTooltip(
	dot: RadarDot,
	opacity: number,
	color: { r: number; g: number; b: number },
): void {
	const tooltipOffset: number = 15;
	const margin: number = 15;
	const tooltipX: number = dot.canvasX + tooltipOffset;
	const tooltipY: number = dot.canvasY - tooltipOffset;
	const padding: number = 10;
	const lineHeight: number = 16;

	// Get current debug configuration
	const config = getTooltipConfig();

	// Don't draw tooltip if tooltips are disabled
	if (!config.show_tooltips) {
		return;
	}

	// Prepare text based on debug configuration
	const texts: string[] = [];

	if (config.track_id) {
		texts.push(`ID: ${dot.track_id || "?"}`);
	}
	if (config.class) {
		texts.push(`Class: ${dot.class || "?"}`);
	}
	if (config.range) {
		texts.push(`Range: ${dot.range ? `${dot.range.toFixed(2)}m` : "?"}`);
	}
	if (config.azimuth) {
		texts.push(
			`Azimuth: ${
				dot.azimuth !== undefined ? `${dot.azimuth.toFixed(1)}°` : "?"
			}`,
		);
	}
	if (config.timestamp) {
		texts.push(`Time: ${dot.timestamp || "?"}`);
	}

	// Don't draw tooltip if no fields are enabled
	if (texts.length === 0) {
		return;
	}

	// Set font for measurement
	ctx.font = "14px monospace";

	// Calculate tooltip dimensions
	const maxWidth: number = Math.max(
		...texts.map((t) => ctx.measureText(t).width),
	);
	const tooltipWidth: number = maxWidth + padding * 2;
	const tooltipHeight: number = texts.length * lineHeight + padding * 2;

	// Adjust tooltip position to keep it within canvas bounds
	let finalTooltipX = tooltipX;
	let finalTooltipY = tooltipY;

	// Prefer showing tooltip to the right of the dot unless it would overflow.
	if (finalTooltipX + tooltipWidth + margin > width) {
		finalTooltipX = dot.canvasX - tooltipWidth - tooltipOffset;
	}

	// If left of the screen, clamp to the margin.
	if (finalTooltipX < margin) {
		finalTooltipX = margin;
	}

	// Prefer showing tooltip above the dot unless it would overflow off-screen.
	if (finalTooltipY < margin) {
		finalTooltipY = dot.canvasY + tooltipOffset;
	}

	// Keep the tooltip entirely inside the canvas bounds after repositioning.
	if (finalTooltipY + tooltipHeight + margin > height) {
		finalTooltipY = height - tooltipHeight - margin;
	}
	if (finalTooltipY < margin) {
		finalTooltipY = margin;
	}

	// Keep tooltip inside the radar circle to avoid being clipped by the bezel overlay.
	const halfWidth = tooltipWidth / 2;
	const halfHeight = tooltipHeight / 2;
	const rectRadius = Math.sqrt(halfWidth * halfWidth + halfHeight * halfHeight);
	const safeRadius = Math.max(0, maxRadius - margin);
	let rectCenterX = finalTooltipX + halfWidth;
	let rectCenterY = finalTooltipY + halfHeight;
	const dxFromCenter = rectCenterX - centerX;
	const dyFromCenter = rectCenterY - centerY;
	const centerDistance = Math.sqrt(
		dxFromCenter * dxFromCenter + dyFromCenter * dyFromCenter,
	);
	const maxCenterDistance = Math.max(0, safeRadius - rectRadius);
	if (centerDistance > maxCenterDistance && centerDistance !== 0) {
		const scale = maxCenterDistance / centerDistance;
		rectCenterX = centerX + dxFromCenter * scale;
		rectCenterY = centerY + dyFromCenter * scale;
		finalTooltipX = rectCenterX - halfWidth;
		finalTooltipY = rectCenterY - halfHeight;
	}

	// Draw tooltip background
	ctx.fillStyle = `rgba(10, 14, 39, ${0.9 * opacity})`;
	ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${
		0.5 * opacity
	})`;
	ctx.lineWidth = 1;
	ctx.beginPath();
	ctx.roundRect(finalTooltipX, finalTooltipY, tooltipWidth, tooltipHeight, 3);
	ctx.fill();
	ctx.stroke();

	// Draw text lines
	ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
	ctx.font = "14px monospace";
	texts.forEach((text, i) => {
		ctx.fillText(
			text,
			finalTooltipX + padding,
			finalTooltipY + padding + lineHeight * (i + 1) - 2,
		);
	});
}

// Draw fading trails for all tracks
export function drawRadarTrails(trackHistory: Map<number, RadarDot[]>): void {
	trackHistory.forEach((history) => {
		if (history.length < 2) return; // Need at least 2 points for a trail

		// Get color based on radar_id (use the most recent dot's radar_id)
		const latestDot = history[history.length - 1];
		const color = getColorForId(latestDot?.radar_id || 0);

		// Draw trail as connected line segments with fading opacity
		for (let i = 0; i < history.length - 1; i++) {
			const current = history[i];
			const next = history[i + 1];
			if (!current || !next) continue;

			// Calculate opacity based on position in history (older = more transparent)
			const progress = i / Math.max(1, history.length - 1);
			const opacity = 0.1 + progress * 0.4; // Range from 0.1 to 0.5

			// Draw line segment
			// ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
			// ctx.lineWidth = 2;
			// ctx.beginPath();
			// ctx.moveTo(current?.canvasX || 0, current?.canvasY || 0);
			// ctx.lineTo(next?.canvasX || 0, next?.canvasY || 0);
			// ctx.stroke();

			// Draw small dots at each historical position
			ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
			ctx.beginPath();
			ctx.arc(current.canvasX, current.canvasY, 2, 0, Math.PI * 2);
			ctx.fill();
		}
	});
}

// Draw all radar dots
export function drawRadarDots(radarDots: RadarDot[]): void {
	radarDots.forEach((dot, index) => {
		// Calculate opacity based on age (newer dots are brighter)
		const age: number = radarDots.length - index;
		const opacity: number = Math.max(0.1, 1 - age / MAX_DOTS);

		// Get color based on radar_id
		const color = getColorForId(dot.radar_id || 0);
		const colorString = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;

		// Draw dot
		ctx.fillStyle = colorString;
		ctx.beginPath();
		ctx.arc(dot.canvasX, dot.canvasY, 7, 0, Math.PI * 2);
		ctx.fill();

		// Add glow effect to newest dots
		if (index === radarDots.length - 1) {
			ctx.shadowBlur = 15;
			ctx.shadowColor = `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`;
			ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 1)`;
			ctx.beginPath();
			ctx.arc(dot.canvasX, dot.canvasY, 7, 0, Math.PI * 2);
			ctx.fill();
			ctx.shadowBlur = 0;
		}

		// Draw tooltip for each dot
		drawDotTooltip(dot, opacity, color);
	});
}

// Update sweep line angle based on time
export function updateSweepLine(healthCheckInterval: number): void {
	const now = Date.now();
	const deltaTime = now - lastSweepUpdate;
	lastSweepUpdate = now;

	// Calculate rotation speed: one full rotation per health check interval
	// Speed in radians per millisecond
	const rotationSpeed = (-2 * Math.PI) / healthCheckInterval;

	// Update angle
	sweepAngle += rotationSpeed * deltaTime;

	// Keep angle in [0, 2π] range
	if (sweepAngle > 2 * Math.PI) {
		sweepAngle -= 2 * Math.PI;
	}
}

// Draw the sweep line
export function drawSweepLine(): void {
	// Save context state
	ctx.save();

	// Move to center
	ctx.translate(centerX, centerY);

	// Rotate to current sweep angle (starting from top, going clockwise)
	// Note: Canvas rotation is clockwise, and 0 radians is to the right
	// We subtract π/2 to start from the top
	ctx.rotate(sweepAngle - Math.PI / 2);

	// Create gradient for the sweep line (fading effect)
	const gradient = ctx.createLinearGradient(0, 0, 0, -maxRadius);
	gradient.addColorStop(0, "rgba(0, 255, 255, 0.1)"); // Center: faint
	gradient.addColorStop(0.7, "rgba(0, 255, 255, 0.4)"); // Mid: moderate
	gradient.addColorStop(1, "rgba(0, 255, 255, 0.8)"); // Edge: bright

	// Draw the sweep line
	ctx.strokeStyle = gradient;
	ctx.lineWidth = 2;
	ctx.beginPath();
	ctx.moveTo(0, 0);
	ctx.lineTo(0, -maxRadius);
	ctx.stroke();

	// Draw a brighter glow at the edge
	ctx.shadowBlur = 10;
	ctx.shadowColor = "rgba(0, 255, 255, 0.6)";
	ctx.strokeStyle = "rgba(0, 255, 255, 0.6)";
	ctx.lineWidth = 1;
	ctx.beginPath();
	ctx.moveTo(0, -maxRadius * 0.9);
	ctx.lineTo(0, -maxRadius);
	ctx.stroke();

	// Restore context state
	ctx.restore();
}

// Draw inactive radar coverage areas
export function drawInactiveRadarAreas(
	radarStatuses: RadarsStatusResponse,
): void {
	// Iterate through all radar statuses
	for (const [_, status] of Object.entries(radarStatuses)) {
		// Only draw greyed area for inactive radars
		if (!status.is_active) {
			const orientationAngle = status.orientation_angle;
			const coverageSpan = 35; // ±35 degrees

			// Calculate start and end angles in radar coordinates
			// Radar orientation: 0° points North (positive Y), increases clockwise
			const radarStartAngle = orientationAngle - coverageSpan;
			const radarEndAngle = orientationAngle + coverageSpan;

			// Convert to canvas radians (0° = East, angles increase clockwise on canvas)
			const canvasStartAngle = radarAngleToCanvas(radarStartAngle);
			const canvasEndAngle = radarAngleToCanvas(radarEndAngle);

			// Draw the greyed coverage wedge
			ctx.save();
			ctx.fillStyle = "rgba(100, 100, 100, 0.6)"; // Grey, semi-transparent
			ctx.beginPath();
			ctx.moveTo(centerX, centerY); // Start at center
			ctx.arc(
				centerX,
				centerY,
				maxRadius,
				canvasStartAngle,
				canvasEndAngle,
				false,
			);
			ctx.closePath();
			ctx.fill();

			// Draw border for the wedge
			ctx.strokeStyle = "rgba(150, 150, 150, 0.5)";
			ctx.lineWidth = 1;
			ctx.stroke();
			ctx.restore();
		}
	}
}

// Draw pulsating center dot with expanding rings
export function drawPulsatingCenter(): void {
	const time = Date.now();
	const pulseDuration = 2500; // Duration of one complete pulse in milliseconds
	const maxRingRadius = 15; // Maximum radius before ring fades completely
	const numberOfRings = 1; // Number of concurrent rings

	// Draw multiple expanding rings at different stages
	for (let i = 0; i < numberOfRings; i++) {
		// Offset each ring's timing so they're evenly spaced
		const ringOffset = (pulseDuration / numberOfRings) * i;
		const ringProgress = ((time + ringOffset) % pulseDuration) / pulseDuration; // 0 to 1

		// Calculate ring radius (expands from center)
		const ringRadius = ringProgress * maxRingRadius;

		// Calculate opacity (fades out as it expands)
		const opacity = (1 - ringProgress) * 0.6; // Start at 0.6, fade to 0

		// Don't draw if too faint
		if (opacity < 0.05) continue;

		// Draw ring
		ctx.strokeStyle = `rgba(0, 255, 255, ${opacity})`;
		ctx.lineWidth = 2;
		ctx.shadowBlur = 10 * (1 - ringProgress); // Blur fades as ring expands
		ctx.shadowColor = `rgba(0, 255, 255, ${opacity * 0.5})`;
		ctx.beginPath();
		ctx.arc(centerX, centerY, ringRadius, 0, Math.PI * 2);
		ctx.stroke();
	}

	// Reset shadow
	ctx.shadowBlur = 0;

	// Draw static center point with glow effect
	// Outer glow
	ctx.shadowBlur = 50;
	ctx.shadowColor = "rgba(0, 255, 255, 0.8)";
	ctx.fillStyle = "rgba(0, 255, 255, 0.3)";
	ctx.beginPath();
	ctx.arc(centerX, centerY, 15, 0, Math.PI * 2);
	ctx.fill();

	// Main center dot (bold and bright)
	ctx.shadowBlur = 30;
	ctx.fillStyle = "rgba(0, 255, 255, 1)";
	ctx.beginPath();
	ctx.arc(centerX, centerY, 6, 0, Math.PI * 2);
	ctx.fill();

	// Reset shadow
	ctx.shadowBlur = 0;
}
