import { beforeEach, describe, expect, test } from "bun:test";
import {
	getLanguage,
	initI18n,
	resetI18nForTests,
	setLanguage,
	subscribeLanguageChange,
	t,
} from "./index.ts";

class MemoryStorage implements Storage {
	#data = new Map<string, string>();

	get length(): number {
		return this.#data.size;
	}

	clear(): void {
		this.#data.clear();
	}

	getItem(key: string): string | null {
		return this.#data.get(key) ?? null;
	}

	key(index: number): string | null {
		return Array.from(this.#data.keys())[index] ?? null;
	}

	removeItem(key: string): void {
		this.#data.delete(key);
	}

	setItem(key: string, value: string): void {
		this.#data.set(key, value);
	}
}

describe("i18n", () => {
	beforeEach(() => {
		resetI18nForTests();
	});

	test("loads language from storage on init", () => {
		const storage = new MemoryStorage();
		storage.setItem("app-language", "he");

		initI18n(storage);

		expect(t("tabs.radar")).toBe('מכ"ם');
		expect(t("debug.menuLabel")).toBe("תפריט ניפוי");
	});

	test("defaults to hebrew when no storage value exists", () => {
		const storage = new MemoryStorage();
		initI18n(storage);

		expect(getLanguage()).toBe("he");
		expect(t("tabs.cameras")).toBe("מצלמות");
	});

	test("persists language change and notifies subscribers", () => {
		const storage = new MemoryStorage();
		initI18n(storage);

		const seen: string[] = [];
		const unsubscribe = subscribeLanguageChange((language) => {
			seen.push(language);
		});

		setLanguage("en");
		unsubscribe();
		setLanguage("he");

		expect(storage.getItem("app-language")).toBe("he");
		expect(seen).toEqual(["en"]);
	});

	test("supports parameterized translations", () => {
		const storage = new MemoryStorage();
		initI18n(storage);

		expect(t("hud.radarsDown", { count: 2 })).toBe('2 מכ"מים מושבתים');
		setLanguage("en");
		expect(t("hud.radarsDown", { count: 2 })).toBe("2 RADARS DOWN");
		setLanguage("he");
		expect(t("hud.radarsDown", { count: 1 })).toBe('מכ"ם 1 מושבת');
	});
});
