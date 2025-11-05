import type { SensorStatus } from "../../types.ts";
import { CORS_HEADERS } from "../config.ts";

// Sensor names
const SENSOR_NAMES = ["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"];

// Health check endpoint handler
export function handleHealth(): Response {
	// Generate status for 4 sensors
	// Each sensor has 80% chance of being healthy
	const sensors: SensorStatus[] = SENSOR_NAMES.map((name, index) => ({
		id: index + 1,
		name,
		healthy: Math.random() > 0.2,
		lastUpdate: Date.now(),
	}));

	// Overall system is healthy if all sensors are healthy
	const overallHealthy = sensors.every((sensor) => sensor.healthy);

	return new Response(
		JSON.stringify({
			sensors,
			overallHealthy,
		}),
		{ headers: CORS_HEADERS }
	);
}
