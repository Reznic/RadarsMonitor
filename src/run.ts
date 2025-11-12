#!/usr/bin/env bun

// Unified launcher - starts both backend and frontend servers
import { type ChildProcess, spawn } from "node:child_process";
import { join } from "node:path";

// Parse command line arguments
const args = process.argv.slice(2);
const isProd = args.includes("--prod") || args.includes("--production");
const mode = isProd ? "production" : "development";

// Set environment
process.env.NODE_ENV = mode;

interface Colors {
	reset: string;
	bright: string;
	cyan: string;
	yellow: string;
	green: string;
	red: string;
	magenta: string;
}

const colors: Colors = {
	reset: "\x1b[0m",
	bright: "\x1b[1m",
	cyan: "\x1b[36m",
	yellow: "\x1b[33m",
	green: "\x1b[32m",
	red: "\x1b[31m",
	magenta: "\x1b[35m",
};

function log(prefix: string, color: string, message: string): void {
	console.log(`${color}${prefix}${colors.reset} ${message}`);
}

// Display startup banner
console.log("\n");
log("[RADAR]", colors.magenta, "━".repeat(60));
log(
	"[RADAR]",
	colors.bright,
	`Starting in ${colors.magenta}${mode.toUpperCase()}${colors.reset} mode`,
);
log("[RADAR]", colors.magenta, "━".repeat(60));
console.log("\n");

// Start Python backend server (using venv)
//const backend: ChildProcess = spawn(
//	"./venv/bin/python",
//	["src/radar_tracks_server.py"],
//	{
//		cwd: process.cwd(),
//		stdio: "pipe",
//		env: { ...process.env, ENVIRONMENT: mode },
//	},
//);
//
//backend.stdout?.on("data", (data: Buffer) => {
//	const lines: string[] = data.toString().trim().split("\n");
//	for (const line of lines) {
//		log("[BACKEND]", colors.cyan, line);
//	}
//});
//
//backend.stderr?.on("data", (data: Buffer) => {
//	const lines: string[] = data.toString().trim().split("\n");
//	for (const line of lines) {
//		log("[BACKEND]", colors.red, line);
//	}
//});
//
//backend.on("close", (code: number | null) => {
//	log("[BACKEND]", colors.red, `Process exited with code ${code}`);
//	frontend.kill();
//	process.exit(code || 0);
//});

// Start frontend server
const frontend: ChildProcess = spawn("bun", ["serve.ts"], {
	cwd: join(process.cwd(), "src/fe"),
	stdio: "pipe",
	env: { ...process.env, NODE_ENV: mode },
});

frontend.stdout?.on("data", (data: Buffer) => {
	const lines: string[] = data.toString().trim().split("\n");
	for (const line of lines) {
		log("[FRONTEND]", colors.yellow, line);
	}
});

frontend.stderr?.on("data", (data: Buffer) => {
	const lines: string[] = data.toString().trim().split("\n");
	for (const line of lines) {
		log("[FRONTEND]", colors.red, line);
	}
});

frontend.on("close", (code: number | null) => {
	log("[FRONTEND]", colors.red, `Process exited with code ${code}`);
	backend.kill();
	process.exit(code || 0);
});

// Handle Ctrl+C gracefully
process.on("SIGINT", () => {
	console.log("\n");
	log("[RADAR]", colors.green, "Shutting down servers...");
	backend.kill();
	frontend.kill();
	process.exit(0);
});

log("[RADAR]", colors.green, `Servers starting...`);
log(
	"[RADAR]",
	colors.bright,
	`Press ${colors.red}Ctrl+C${colors.reset} to stop both servers`,
);
console.log("\n");
