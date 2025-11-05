// Server configuration
export const PORT: number = 1337;

// Environment mode - use function for runtime evaluation (important for testing)
export const IS_DEV_MODE = (): boolean => process.env.NODE_ENV !== "production";

// Legacy constant export for backward compatibility
export const IS_DEV = IS_DEV_MODE();

// CORS configuration
export const CORS_HEADERS: Record<string, string> = {
	"Access-Control-Allow-Origin": "*",
	"Access-Control-Allow-Methods": "GET, OPTIONS",
	"Access-Control-Allow-Headers": "Content-Type",
	"Content-Type": "application/json",
} as const;
