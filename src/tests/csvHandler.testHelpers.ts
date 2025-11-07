/**
 * Test helper functions for csvHandler
 * These functions provide controlled access to csvHandler's internal state for testing
 */

// Import state accessors that will be exported from csvHandler
import {
	__internal_getState,
	__internal_setState,
	stopPlayback,
} from "../be/data/csvHandler.ts";
import type { CSVRow } from "../types.ts";

export interface CSVHandlerTestState {
	csvData: CSVRow[];
	currentTimestampIndex: number;
	timestamps: number[];
	isPlaying: boolean;
	lastReadPosition: number;
	csvPath: string;
	csvHeaders: string[];
	consecutiveErrors: number;
}

/**
 * Reset all csvHandler state - useful for test isolation
 */
export function __test_reset(): void {
	// Stop any running playback/watchers
	stopPlayback();

	// Reset all state
	__internal_setState({
		csvData: [],
		currentTimestampIndex: 0,
		timestamps: [],
		isPlaying: false,
		lastReadPosition: 0,
		csvPath: "",
		csvHeaders: [],
		consecutiveErrors: 0,
	});
}

/**
 * Get a snapshot of current csvHandler state for assertions
 */
export function __test_getState(): CSVHandlerTestState {
	return __internal_getState();
}
