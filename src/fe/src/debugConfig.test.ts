import { describe, expect, test } from "bun:test";
import {
	getAvailableFields,
	isCameraUrlsVisible,
	setCameraUrlsVisible,
	subscribeDebugConfig,
} from "./debugConfig.ts";

describe("debugConfig camera URL visibility", () => {
	test("includes show_camera_urls field and defaults to disabled", () => {
		const fields = getAvailableFields();
		const cameraField = fields.find(
			(field) => field.key === "show_camera_urls",
		);

		expect(cameraField).toBeDefined();
		expect(cameraField?.enabled).toBe(false);
		expect(isCameraUrlsVisible()).toBe(false);
	});

	test("notifies subscribers when camera URL visibility changes", () => {
		const notifications: boolean[] = [];
		const unsubscribe = subscribeDebugConfig((config) => {
			notifications.push(config.show_camera_urls);
		});

		setCameraUrlsVisible(false);
		setCameraUrlsVisible(true);
		unsubscribe();
		setCameraUrlsVisible(false);

		expect(notifications).toEqual([false, true]);
		expect(isCameraUrlsVisible()).toBe(false);
	});
});
