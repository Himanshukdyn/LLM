const runtime = (typeof window !== "undefined" && window.__APP_CONFIG__) || {};
const buildTime = (import.meta.env && import.meta.env.VITE_API_BASE_URL) || "";

export const API_BASE_URL = (runtime.API_BASE_URL || buildTime || "http://localhost:8000").replace(
  /\/+$/,
  ""
);

export function newSessionId() {
  const random =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
      : Math.random().toString(36).slice(2, 14);
  return `s-${random}`;
}

/** Turn a FastAPI error body into one readable sentence. */
function describeError(status, body) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d) => (d.msg || "invalid input").replace(/^Value error, /, ""))
      .join("; ");
  }
  return `request failed with status ${status}`;
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/**
 * POST /chat. Never throws anything other than ApiError, so the UI can always
 * render a readable message instead of a blank screen.
 */
export async function sendChat({ sessionId, message, systemPrompt, simulateFailure }) {
  const headers = { "Content-Type": "application/json" };
  if (simulateFailure) headers["X-Simulate-Failure"] = simulateFailure;
  const payload = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) payload.system_prompt = systemPrompt.trim();

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(0, "Could not reach the server. Check your connection and try again.");
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    /* non-JSON body: handled below */
  }
  if (!response.ok) throw new ApiError(response.status, describeError(response.status, body));
  if (!body || typeof body.reply !== "string" || !body.metadata) {
    throw new ApiError(response.status, "The server sent an unexpected response.");
  }
  return body;
}
