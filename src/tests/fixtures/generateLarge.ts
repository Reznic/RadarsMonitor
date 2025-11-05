#!/usr/bin/env bun
// Script to generate a large CSV file for performance testing
import { writeFileSync } from "node:fs";
import { join } from "node:path";

const OUTPUT_FILE = join(import.meta.dir, "large.csv");
const NUM_TRACKS = 1000; // Number of unique tracks
const SAMPLES_PER_TRACK = 100; // Samples per track

const header =
	"track_id,timestamp,Radar id,time_from_start,num assoc,range,doppler,x,y,z,vx,vy,vz,calc_dopp,is assoc,assoc_timestamp,assoc_x,assoc_y,assoc_z,assoc_dopp,z var,class\n";

let csv = header;
let lineCount = 0;

for (let trackId = 1; trackId <= NUM_TRACKS; trackId++) {
	const startX = Math.random() * 100 - 50;
	const startY = Math.random() * 100 - 50;
	const vx = (Math.random() - 0.5) * 2;
	const vy = (Math.random() - 0.5) * 2;
	const classType = ["Human", "Vehicle", "Bird", "Drone"][
		Math.floor(Math.random() * 4)
	];

	for (let sample = 0; sample < SAMPLES_PER_TRACK; sample++) {
		const timestamp = (sample * 0.05).toFixed(2);
		const x = (startX + vx * sample).toFixed(1);
		const y = (startY + vy * sample).toFixed(1);
		const z = (Math.random() * 2 - 1).toFixed(1);
		const range = Math.sqrt(Number(x) ** 2 + Number(y) ** 2).toFixed(1);
		const doppler = (Math.random() * 2 - 1).toFixed(1);

		csv += `${trackId},${timestamp},0,${timestamp},${sample},${range},${doppler},${x},${y},${z},${vx.toFixed(1)},${vy.toFixed(1)},0.0,${doppler},1,${timestamp},${x},${y},${z},${doppler},1.0,${classType}\n`;
		lineCount++;
	}
}

writeFileSync(OUTPUT_FILE, csv);

console.log(
	`Generated large.csv with ${NUM_TRACKS} tracks, ${SAMPLES_PER_TRACK} samples per track = ${lineCount} total lines`,
);
console.log(`File size: ${(csv.length / 1024 / 1024).toFixed(2)} MB`);
