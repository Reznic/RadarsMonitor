import { CORS_HEADERS, IS_DEV_MODE, PORT } from "./config.ts";
import { loadCSV } from "./data/csvHandler.ts";
import { handleHealth } from "./routes/health.ts";
import { handleRadar } from "./routes/radar.ts";

// Load CSV data on startup
console.log(`📊 Loading CSV data in ${IS_DEV_MODE() ? "DEV" : "PROD"} mode...`);
loadCSV();

const server = Bun.serve({
	port: PORT,

	fetch(req: Request): Response {
		const url: URL = new URL(req.url);

		// Handle CORS preflight
		if (req.method === "OPTIONS") {
			return new Response(null, { headers: CORS_HEADERS });
		}

		// Route handlers
		if (url.pathname === "/health") {
			return handleHealth();
		}

		if (url.pathname === "/radar") {
			return handleRadar();
		}

		// 404 for other routes
		return new Response("Not Found", { status: 404 });
	},
});

console.log(`🎯 Radar server running on http://localhost:${server.port}`);
console.log(`   Health endpoint: http://localhost:${server.port}/health`);
console.log(`   Radar endpoint:  http://localhost:${server.port}/radar`);
