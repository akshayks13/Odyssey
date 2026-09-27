/** A thread id for the URL. `crypto.randomUUID` needs a secure context (https, localhost or
 * 127.0.0.1) — on a plain http LAN address (e.g. opening the app from a phone at
 * http://192.168.x.x:3000, which next.config.mjs's own comments say should work) it's
 * `undefined` and calling it throws inside the Search button's click handler, so nothing
 * happens with no visible error. This falls back to a good-enough random id instead. */
export function newThreadId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "id-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}
