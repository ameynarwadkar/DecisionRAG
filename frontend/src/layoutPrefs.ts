export const LAYOUT_STORAGE_KEY = "dec_rag_layout_prefs_v1";
export const MIN_ARCH_HEIGHT = 90;
export const MAX_ARCH_HEIGHT = 560;
export const ARCH_ASPECT_RATIO = 1040 / 226;

export interface LayoutPreferences {
  leftCollapsed: boolean;
  rightCollapsed: boolean;
  archHeight: number | null;
}

function defaultRightCollapsed(): boolean {
  if (typeof window !== "undefined" && typeof window.innerWidth === "number") {
    return window.innerWidth <= 1080;
  }
  return false;
}

export function clampArchHeight(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  const rounded = Math.round(value);
  return Math.min(MAX_ARCH_HEIGHT, Math.max(MIN_ARCH_HEIGHT, rounded));
}

export function loadLayoutPrefs(): LayoutPreferences {
  const defaults: LayoutPreferences = {
    leftCollapsed: false,
    rightCollapsed: defaultRightCollapsed(),
    archHeight: null,
  };
  if (typeof localStorage === "undefined") {
    return defaults;
  }
  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw) as Partial<
      Record<keyof LayoutPreferences, unknown>
    >;
    return {
      leftCollapsed:
        typeof parsed.leftCollapsed === "boolean"
          ? parsed.leftCollapsed
          : defaults.leftCollapsed,
      rightCollapsed:
        typeof parsed.rightCollapsed === "boolean"
          ? parsed.rightCollapsed
          : defaults.rightCollapsed,
      archHeight:
        parsed.archHeight === null ? null : clampArchHeight(parsed.archHeight),
    };
  } catch {
    return defaults;
  }
}

export function updateLayoutPrefs(
  patch: Partial<LayoutPreferences>
): LayoutPreferences {
  const current = loadLayoutPrefs();
  const next: LayoutPreferences = {
    leftCollapsed:
      patch.leftCollapsed !== undefined
        ? patch.leftCollapsed
        : current.leftCollapsed,
    rightCollapsed:
      patch.rightCollapsed !== undefined
        ? patch.rightCollapsed
        : current.rightCollapsed,
    archHeight:
      patch.archHeight === undefined
        ? current.archHeight
        : patch.archHeight === null
          ? null
          : clampArchHeight(patch.archHeight),
  };
  if (typeof localStorage !== "undefined") {
    try {
      localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Ignore storage quota / privacy errors
    }
  }
  return next;
}
