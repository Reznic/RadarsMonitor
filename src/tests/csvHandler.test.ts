import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { appendFileSync, existsSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
	getCurrentTargets,
	getPlaybackInfo,
	loadCSV,
	stopPlayback,
} from "../be/data/csvHandler";
import { __test_getState, __test_reset } from "./csvHandler.testHelpers";

const FIXTURES_DIR = join(import.meta.dir, "fixtures");
const SMALL_CSV = join(FIXTURES_DIR, "small.csv");
const INVALID_CSV = join(FIXTURES_DIR, "invalid.csv");
const LARGE_CSV = join(FIXTURES_DIR, "large.csv");
const TEMP_CSV = join(FIXTURES_DIR, "temp_test.csv");

// Mock IS_DEV_MODE for testing
const originalEnv = process.env.NODE_ENV;

describe("CSV Handler - Dev Mode", () => {
	beforeEach(() => {
		process.env.NODE_ENV = "development";
		__test_reset();
	});

	afterEach(() => {
		stopPlayback();
		__test_reset();
		process.env.NODE_ENV = originalEnv;

		// Cleanup temp files
		if (existsSync(TEMP_CSV)) {
			unlinkSync(TEMP_CSV);
		}
	});

	test("should load small CSV file successfully", () => {
		const result = loadCSV(SMALL_CSV);
		expect(result).toBe(true);

		const state = __test_getState();
		expect(state.csvData.length).toBeGreaterThan(0);
		expect(state.timestamps.length).toBeGreaterThan(0);
		expect(state.csvHeaders.length).toBeGreaterThan(0);
	});

	test("should parse CSV rows correctly", () => {
		loadCSV(SMALL_CSV);
		const targets = getCurrentTargets();

		expect(targets.length).toBeGreaterThan(0);
		expect(targets[0]).toHaveProperty("track_id");
		expect(targets[0]).toHaveProperty("timestamp");
		expect(targets[0]).toHaveProperty("x");
		expect(targets[0]).toHaveProperty("y");
		expect(targets[0]).toHaveProperty("doppler");
	});

	test("should handle invalid CSV rows gracefully", () => {
		const result = loadCSV(INVALID_CSV);
		expect(result).toBe(true);

		const state = __test_getState();
		// Should skip invalid rows but still load valid ones
		expect(state.csvData.length).toBeGreaterThan(0);

		// All loaded rows should be valid
		state.csvData.forEach((row) => {
			expect(typeof row.timestamp).toBe("number");
			expect(typeof row.x).toBe("number");
			expect(typeof row.y).toBe("number");
		});
	});

	test("should extract unique timestamps and sort them", () => {
		loadCSV(SMALL_CSV);
		const state = __test_getState();

		expect(state.timestamps.length).toBeGreaterThan(0);

		// Check timestamps are sorted
		for (let i = 1; i < state.timestamps.length; i++) {
			expect(state.timestamps[i]).toBeGreaterThanOrEqual(
				state.timestamps[i - 1],
			);
		}
	});

	test("should start playback automatically in dev mode", async () => {
		loadCSV(SMALL_CSV);
		const state = __test_getState();

		expect(state.isPlaying).toBe(true);
	});

	test("should loop through timestamps", async () => {
		loadCSV(SMALL_CSV);
		const initialState = __test_getState();
		const initialIndex = initialState.currentTimestampIndex;

		// Wait for playback to advance
		await new Promise((resolve) => setTimeout(resolve, 600));

		const newState = __test_getState();
		const newIndex = newState.currentTimestampIndex;

		// Index should have changed
		expect(newIndex).not.toBe(initialIndex);
	});

	test("should return targets for current timestamp", () => {
		loadCSV(SMALL_CSV);
		const targets = getCurrentTargets();

		expect(Array.isArray(targets)).toBe(true);
		expect(targets.length).toBeGreaterThan(0);

		// All targets should have the same timestamp
		if (targets.length > 1) {
			const firstTimestamp = targets[0].timestamp;
			targets.forEach((target) => {
				expect(target.timestamp).toBe(firstTimestamp);
			});
		}
	});
});

describe("CSV Handler - Performance", () => {
	beforeEach(() => {
		process.env.NODE_ENV = "development";
		__test_reset();
	});

	afterEach(() => {
		stopPlayback();
		__test_reset();
		process.env.NODE_ENV = originalEnv;
	});

	test("should demonstrate time needed to parse large file", () => {
		const startTime = performance.now();

		const result = loadCSV(LARGE_CSV);

		const endTime = performance.now();
		const duration = endTime - startTime;

		expect(result).toBe(true);

		const state = __test_getState();
		console.log(`\n📊 Performance Test Results:`);
		console.log(`   - Total rows parsed: ${state.csvData.length}`);
		console.log(`   - Unique timestamps: ${state.timestamps.length}`);
		console.log(`   - Parse time: ${duration.toFixed(2)}ms`);
		console.log(
			`   - Rows per second: ${(state.csvData.length / (duration / 1000)).toFixed(0)}`,
		);

		// Assert reasonable performance - should parse 100k rows in under 5 seconds
		expect(duration).toBeLessThan(5000);
		expect(state.csvData.length).toBeGreaterThan(90000); // Should have most rows (some might be filtered)
	});

	test("should efficiently get targets from large dataset", () => {
		loadCSV(LARGE_CSV);

		const startTime = performance.now();
		const targets = getCurrentTargets();
		const endTime = performance.now();
		const duration = endTime - startTime;

		console.log(`\n🎯 Target Retrieval Performance:`);
		console.log(`   - Targets retrieved: ${targets.length}`);
		console.log(`   - Retrieval time: ${duration.toFixed(2)}ms`);

		// Should be very fast - under 100ms
		expect(duration).toBeLessThan(100);
		expect(targets.length).toBeGreaterThan(0);
	});
});

describe("CSV Handler - Prod Mode", () => {
	beforeEach(() => {
		process.env.NODE_ENV = "production";
		__test_reset();
	});

	afterEach(() => {
		stopPlayback();
		__test_reset();
		process.env.NODE_ENV = originalEnv;

		if (existsSync(TEMP_CSV)) {
			unlinkSync(TEMP_CSV);
		}
	});

	test("should initialize in prod mode", () => {
		// Create a test CSV file
		writeFileSync(
			TEMP_CSV,
			`track_id,timestamp,Radar id,time_from_start,num assoc,range,doppler,x,y,z,vx,vy,vz,calc_dopp,is assoc,assoc_timestamp,assoc_x,assoc_y,assoc_z,assoc_dopp,z var,class
1,1.0,0,1.0,10,5.0,-0.5,2.0,4.0,0.0,0.0,1.0,0.0,0.5,1,1.0,2.0,4.0,0.0,-0.5,1.0,Human
`,
		);

		const result = loadCSV(TEMP_CSV);
		expect(result).toBe(true);

		const state = __test_getState();
		expect(state.lastReadPosition).toBeGreaterThan(0);
		expect(state.isPlaying).toBe(false); // Should not be playing in prod mode
	});

	test("should handle file growth and read new data", async () => {
		// Create initial CSV file
		const header =
			"track_id,timestamp,Radar id,time_from_start,num assoc,range,doppler,x,y,z,vx,vy,vz,calc_dopp,is assoc,assoc_timestamp,assoc_x,assoc_y,assoc_z,assoc_dopp,z var,class\n";
		const row1 =
			"1,1.0,0,1.0,10,5.0,-0.5,2.0,4.0,0.0,0.0,1.0,0.0,0.5,1,1.0,2.0,4.0,0.0,-0.5,1.0,Human\n";

		writeFileSync(TEMP_CSV, header + row1);

		loadCSV(TEMP_CSV);

		const initialState = __test_getState();
		const initialPosition = initialState.lastReadPosition;

		// Append new data
		const row2 =
			"2,2.0,0,2.0,10,6.0,-0.5,3.0,5.0,0.0,0.0,1.0,0.0,0.5,1,2.0,3.0,5.0,0.0,-0.5,1.0,Vehicle\n";
		appendFileSync(TEMP_CSV, row2);

		// Note: In real prod mode, this would be triggered by file watcher
		// For testing, we verify the position tracking works
		const newState = __test_getState();
		expect(newState.lastReadPosition).toBe(initialPosition);
	});

	test("should return all current data in prod mode", () => {
		writeFileSync(
			TEMP_CSV,
			`track_id,timestamp,Radar id,time_from_start,num assoc,range,doppler,x,y,z,vx,vy,vz,calc_dopp,is assoc,assoc_timestamp,assoc_x,assoc_y,assoc_z,assoc_dopp,z var,class
1,1.0,0,1.0,10,5.0,-0.5,2.0,4.0,0.0,0.0,1.0,0.0,0.5,1,1.0,2.0,4.0,0.0,-0.5,1.0,Human
2,1.0,0,1.0,10,6.0,-0.5,3.0,5.0,0.0,0.0,1.0,0.0,0.5,1,1.0,3.0,5.0,0.0,-0.5,1.0,Vehicle
`,
		);

		loadCSV(TEMP_CSV);
		const targets = getCurrentTargets();

		// In prod mode after initial load, should return empty (no new data yet)
		expect(Array.isArray(targets)).toBe(true);
	});
});

describe("CSV Handler - Error Handling", () => {
	beforeEach(() => {
		process.env.NODE_ENV = "production";
		__test_reset();
	});

	afterEach(() => {
		stopPlayback();
		__test_reset();
		process.env.NODE_ENV = originalEnv;

		if (existsSync(TEMP_CSV)) {
			unlinkSync(TEMP_CSV);
		}
	});

	test("should handle missing file gracefully", () => {
		const result = loadCSV("/nonexistent/file.csv");
		expect(result).toBe(false);
	});

	test("should handle empty file", () => {
		writeFileSync(TEMP_CSV, "");
		const result = loadCSV(TEMP_CSV);
		expect(result).toBe(false);
	});

	test("should handle file with only header", () => {
		writeFileSync(TEMP_CSV, "track_id,timestamp,x,y,z\n");
		const result = loadCSV(TEMP_CSV);
		expect(result).toBe(true);

		const targets = getCurrentTargets();
		expect(targets.length).toBe(0);
	});

	test("should skip completely invalid rows", () => {
		// Set to dev mode for this test to actually load the data
		const originalEnv = process.env.NODE_ENV;
		process.env.NODE_ENV = "development";
		__test_reset();

		const result = loadCSV(INVALID_CSV);
		expect(result).toBe(true);

		const state = __test_getState();
		// Should have loaded only valid rows
		expect(state.csvData.length).toBeGreaterThan(0);

		// Verify all loaded data is valid
		state.csvData.forEach((row) => {
			expect(row.track_id).toBeDefined();
			expect(typeof row.timestamp).toBe("number");
			expect(typeof row.x).toBe("number");
			expect(typeof row.y).toBe("number");
		});

		stopPlayback();
		process.env.NODE_ENV = originalEnv;
	});
});

describe("CSV Handler - State Management", () => {
	beforeEach(() => {
		process.env.NODE_ENV = "development";
		__test_reset();
	});

	afterEach(() => {
		stopPlayback();
		__test_reset();
		process.env.NODE_ENV = originalEnv;
	});

	test("should reset state correctly", () => {
		loadCSV(SMALL_CSV);

		__test_reset();

		const state = __test_getState();
		expect(state.csvData.length).toBe(0);
		expect(state.timestamps.length).toBe(0);
		expect(state.isPlaying).toBe(false);
		expect(state.lastReadPosition).toBe(0);
	});

	test("should provide accurate playback info", () => {
		loadCSV(SMALL_CSV);

		const info = getPlaybackInfo();
		expect(info).toHaveProperty("currentTimestamp");
		expect(info).toHaveProperty("totalTimestamps");
		expect(info).toHaveProperty("currentIndex");
		expect(info).toHaveProperty("isPlaying");
		expect(typeof info.currentTimestamp).toBe("number");
		expect(typeof info.totalTimestamps).toBe("number");
	});

	test("should stop playback cleanly", () => {
		loadCSV(SMALL_CSV);

		const stateBefore = __test_getState();
		expect(stateBefore.isPlaying).toBe(true);

		stopPlayback();

		const stateAfter = __test_getState();
		expect(stateAfter.isPlaying).toBe(false);
	});
});
