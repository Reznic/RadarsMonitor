import { enDictionary, heDictionary } from "./dictionaries.ts";

export type AppLanguage = "en" | "he";

type TranslationParams = Record<string, number | string>;
type PluralValue = { one: string; other: string };
type TranslationValue = string | PluralValue;

type EnDictionary = typeof enDictionary;
export type TranslationKey = keyof EnDictionary;

type Dictionary = Record<string, TranslationValue>;

const STORAGE_KEY = "app-language";

const dictionaries: Record<AppLanguage, Dictionary> = {
	en: enDictionary as Dictionary,
	he: heDictionary as Dictionary,
};

let initialized = false;
let currentLanguage: AppLanguage = "he";
let storageRef: Pick<Storage, "getItem" | "setItem"> | null = null;
const listeners = new Set<(language: AppLanguage) => void>();

function getBrowserStorage(): Pick<Storage, "getItem" | "setItem"> | null {
	try {
		if (typeof localStorage === "undefined") {
			return null;
		}
		return localStorage;
	} catch {
		return null;
	}
}

function normalizeLanguage(language: string | null | undefined): AppLanguage {
	return language === "en" ? "en" : "he";
}

function syncDocumentLanguage(): void {
	if (typeof document === "undefined") return;
	document.documentElement.lang = currentLanguage;
	document.documentElement.dir = currentLanguage === "he" ? "rtl" : "ltr";
}

function interpolate(template: string, params: TranslationParams): string {
	return template.replace(/\{\{\s*([\w.-]+)\s*\}\}/g, (_, key: string) => {
		const value = params[key];
		return value !== undefined ? String(value) : "";
	});
}

function resolveTranslationValue(
	value: TranslationValue,
	params: TranslationParams,
): string {
	if (typeof value === "string") {
		return interpolate(value, params);
	}

	const count = Number(params.count ?? 0);
	const template = count === 1 ? value.one : value.other;
	return interpolate(template, params);
}

function hasTranslationKey(key: string): key is TranslationKey {
	return key in dictionaries.en;
}

export function initI18n(
	storage: Pick<Storage, "getItem" | "setItem"> | null = null,
): void {
	if (initialized) return;
	storageRef = storage ?? getBrowserStorage();
	currentLanguage = normalizeLanguage(storageRef?.getItem(STORAGE_KEY));
	initialized = true;
	syncDocumentLanguage();
}

export function getLanguage(): AppLanguage {
	return currentLanguage;
}

export function setLanguage(language: AppLanguage): void {
	const nextLanguage = normalizeLanguage(language);
	if (nextLanguage === currentLanguage) return;
	currentLanguage = nextLanguage;
	syncDocumentLanguage();
	if (storageRef) {
		try {
			storageRef.setItem(STORAGE_KEY, currentLanguage);
		} catch {
			// no-op when storage is unavailable
		}
	}
	for (const listener of listeners) {
		listener(currentLanguage);
	}
}

export function t(key: TranslationKey, params: TranslationParams = {}): string {
	const fallback = dictionaries.en[key as string];
	const current = dictionaries[currentLanguage][key as string] ?? fallback;
	if (!current) return key;
	return resolveTranslationValue(current, params);
}

export function getLocalizedCameraName(name: string, id?: number): string {
	if (currentLanguage !== "he") return name;
	const match = /^camera\s+(\d+)$/i.exec(name.trim());
	const numericId =
		id ??
		(match && Number.isFinite(Number(match[1])) ? Number(match[1]) : null);
	if (numericId !== null && numericId !== undefined) {
		return t("camera.defaultNameWithId", { id: numericId });
	}
	return name;
}

export function subscribeLanguageChange(
	listener: (language: AppLanguage) => void,
): () => void {
	listeners.add(listener);
	return () => {
		listeners.delete(listener);
	};
}

export function applyDocumentTranslations(root: ParentNode = document): void {
	if (!(root as ParentNode).querySelectorAll) return;

	const textNodes = root.querySelectorAll<HTMLElement>("[data-i18n]");
	for (const node of Array.from(textNodes)) {
		const key = node.dataset.i18n;
		if (!key || !hasTranslationKey(key)) continue;
		node.textContent = t(key);
	}

	const attrNodes = root.querySelectorAll<HTMLElement>("[data-i18n-attr]");
	for (const node of Array.from(attrNodes)) {
		const mapping = node.dataset.i18nAttr;
		if (!mapping) continue;
		const pairs = mapping.split(",");
		for (const rawPair of pairs) {
			const [attributeName, key] = rawPair
				.split(":")
				.map((part) => part.trim());
			if (!attributeName || !key || !hasTranslationKey(key)) continue;
			node.setAttribute(attributeName, t(key));
		}
	}

	if (typeof document !== "undefined" && root === document) {
		document.title = t("app.title");
	}
}

export function resetI18nForTests(): void {
	initialized = false;
	currentLanguage = "he";
	listeners.clear();
	storageRef = null;
}
