#!/usr/bin/env bun

// Development launcher - starts both backend and frontend servers
import { type ChildProcess, spawn } from "node:child_process";
import { join } from "node:path";

// Set environment to development
process.env.NODE_ENV = "development";

interface Colors {
	reset: string;
	bright: string;
	cyan: string;
	yellow: string;
	green: string;
	red: string;
}

const colors: Colors = {
	reset: "\x1b[0m",
	bright: "\x1b[1m",
	cyan: "\x1b[36m",
	yellow: "\x1b[33m",
	green: "\x1b[32m",
	red: "\x1b[31m",
};

function log(prefix: string, color: string, message: string): void {
	console.log(`${color}${prefix}${colors.reset} ${message}`);
}

// Start backend server
const backend: ChildProcess = spawn("bun", ["src/be/server.ts"], {
	cwd: process.cwd(),
	stdio: "pipe",
	env: { ...process.env, NODE_ENV: "development" },
});

backend.stdout?.on("data", (data: Buffer) => {
	const lines: string[] = data.toString().trim().split("\n");
	for (const line of lines) {
		log("[BACKEND]", colors.cyan, line);
	}
});

backend.stderr?.on("data", (data: Buffer) => {
	const lines: string[] = data.toString().trim().split("\n");
	for (const line of lines) {
		log("[BACKEND]", colors.red, line);
	}
});

backend.on("close", (code: number | null) => {
	log("[BACKEND]", colors.red, `Process exited with code ${code}`);
	frontend.kill();
	process.exit(code || 0);
});

// Start frontend server
const frontend: ChildProcess = spawn("bun", ["serve.ts"], {
	cwd: join(process.cwd(), "src/fe"),
	stdio: "pipe",
	env: { ...process.env, NODE_ENV: "development" },
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
	log("[DEV]", colors.green, "Shutting down servers...");
	backend.kill();
	frontend.kill();
	process.exit(0);
});

log("[DEV]", colors.green, "Starting development servers...");
log("[DEV]", colors.bright, "Press Ctrl+C to stop both servers");
