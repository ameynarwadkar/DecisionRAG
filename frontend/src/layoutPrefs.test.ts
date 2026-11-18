import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";
import {
  MAX_ARCH_HEIGHT,
  MIN_ARCH_HEIGHT,
  clampArchHeight,
  loadLayoutPrefs,
  updateLayoutPrefs,
} from "./layoutPrefs.ts";

class MemoryStorage {
  private store = new Map<string, string>();
  getItem(k: string): string | null {
    return this.store.has(k) ? this.store.get(k)! : null;
  }
  setItem(k: string, v: string): void {
    this.store.set(k, v);
  }
  clear(): void {
    this.store.clear();
  }
}

const mem = new MemoryStorage();
Object.defineProperty(globalThis, "localStorage", {
  value: mem,
  configurable: true,
});

describe("layoutPrefs", () => {
  beforeEach(() => {
    mem.clear();
  });

  it("returns sensible defaults when storage is empty", () => {
    const prefs = loadLayoutPrefs();
    assert.equal(prefs.leftCollapsed, false);
    assert.equal(prefs.rightCollapsed, false);
    assert.equal(prefs.archHeight, null);
  });

  it("clamps custom architecture heights into [90, 560]", () => {
    assert.equal(clampArchHeight(null), null);
    assert.equal(clampArchHeight("bad"), null);
    assert.equal(clampArchHeight(40), MIN_ARCH_HEIGHT);
    assert.equal(clampArchHeight(999), MAX_ARCH_HEIGHT);
    assert.equal(clampArchHeight(310.4), 310);
  });

  it("merges partial updates without overwriting sibling fields", () => {
    updateLayoutPrefs({ leftCollapsed: true });
    updateLayoutPrefs({ archHeight: 340 });
    const prefs = loadLayoutPrefs();
    assert.equal(prefs.leftCollapsed, true);
    assert.equal(prefs.rightCollapsed, false);
    assert.equal(prefs.archHeight, 340);

    updateLayoutPrefs({ archHeight: null });
    assert.equal(loadLayoutPrefs().archHeight, null);
    assert.equal(loadLayoutPrefs().leftCollapsed, true);
  });

  it("recovers gracefully from corrupted JSON in localStorage", () => {
    mem.setItem("dec_rag_layout_prefs_v1", "{not-valid-json");
    const prefs = loadLayoutPrefs();
    assert.equal(prefs.leftCollapsed, false);
    assert.equal(prefs.archHeight, null);
  });
});
