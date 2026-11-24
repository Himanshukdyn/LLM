import { useEffect, useRef, useState } from "react";
import { API_BASE_URL, ApiError, newSessionId, sendChat } from "./api.js";

const FAILURE_PRESETS = [
  { label: "No simulated failure", value: "" },
  { label: "key_1 → 429 (rate limit)", value: "key_1=429" },
  { label: "key_1 → 500 (server error)", value: "key_1=500" },
  { label: "All keys → 429", value: "*=429" },
];

function MetaItem({ label, testId, children, tone }) {
  return (
    <span className={`meta-item ${tone || ""}`}>
      <span className="meta-label">{label}</span>
      <span className="meta-value" data-testid={testId}>
        {children}
      </span>
    </span>
  );
}

function AssistantMessage({ message }) {
  const m = message.metadata;
  const cached = m.resolution_source === "LLM_CACHE";
  return (
    <div className="row assistant" data-testid="assistant-message">
      <div className="bubble">
        <div className="text">{message.content}</div>
        <div className="meta">
          <MetaItem label="source" testId="meta-source" tone={cached ? "cache" : "live"}>
            {m.resolution_source}
          </MetaItem>
          <MetaItem label="model" testId="meta-model">
            {m.model}
          </MetaItem>
          <MetaItem label="latency" testId="meta-latency">
            {m.latency_ms} ms
          </MetaItem>
          <MetaItem label="in tokens" testId="meta-input-tokens">
            {m.input_tokens}
          </MetaItem>
          <MetaItem label="out tokens" testId="meta-output-tokens">
            {m.output_tokens}
          </MetaItem>
          <MetaItem label="key" testId="meta-key-id">
            {m.key_id ?? "—"}
          </MetaItem>
          {m.fallback_attempts > 0 && (
            <MetaItem label="failovers" testId="meta-fallback-attempts" tone="warn">
              {m.fallback_attempts}
            </MetaItem>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  // A new session on every page load: the id is never persisted.
  const [sessionId, setSessionId] = useState(newSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulateFailure, setSimulateFailure] = useState("");
  const [sending, setSending] = useState(false);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);

  async function send() {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    setSending(true);
    const userId = crypto.randomUUID?.() ?? String(Date.now());
    setMessages((prev) => [...prev, { id: userId, role: "user", content: text }]);
    try {
      const data = await sendChat({ sessionId, message: text, systemPrompt, simulateFailure });
      setMessages((prev) => [
        ...prev,
        { id: `${userId}-a`, role: "assistant", content: data.reply, metadata: data.metadata },
      ]);
    } catch (err) {
      const status = err instanceof ApiError ? err.status : 0;
      const text =
        err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
      setMessages((prev) => [
        // A failed request is not stored by the server, so mark the user turn too.
        ...prev.map((m) => (m.id === userId ? { ...m, failed: true } : m)),
        { id: `${userId}-e`, role: "error", status, content: text },
      ]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  }

  function newChat() {
    setSessionId(newSessionId());
    setMessages([]);
    setInput("");
  }

  const tooLong = input.trim().length > 4000;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>LLM Chatbot</h1>
          <p className="sub">
            Session <code>{sessionId}</code>
          </p>
        </div>
        <button type="button" className="ghost" onClick={newChat} disabled={sending}>
          New chat
        </button>
      </header>

      <details className="settings">
        <summary>Session settings</summary>
        <label className="field">
          <span>System prompt (optional, applies to this session)</span>
          <textarea
            data-testid="system-prompt-input"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="You are a helpful, concise assistant."
            rows={2}
          />
        </label>
        <label className="field">
          <span>
            Failover demo: sends <code>X-Simulate-Failure</code> with the next message
          </span>
          <select
            data-testid="simulate-failure-select"
            value={simulateFailure}
            onChange={(e) => setSimulateFailure(e.target.value)}
          >
            {FAILURE_PRESETS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <p className="hint">
          Backend: <code>{API_BASE_URL}</code>
        </p>
      </details>

      <main className="messages" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask anything to start. Send the same first message in a new chat to see a cache hit.</p>
          </div>
        )}
        {messages.map((m) => {
          if (m.role === "assistant") return <AssistantMessage key={m.id} message={m} />;
          if (m.role === "error")
            return (
              <div key={m.id} className="row error" role="alert" data-testid="error-message">
                <div className="bubble">
                  <strong>{m.status ? `Error ${m.status}` : "Network error"}:</strong> {m.content}
                  <div className="error-note">This message was not saved to the conversation.</div>
                </div>
              </div>
            );
          return (
            <div key={m.id} className={`row user ${m.failed ? "failed" : ""}`} data-testid="user-message">
              <div className="bubble">{m.content}</div>
            </div>
          );
        })}
        {sending && (
          <div className="row assistant pending">
            <div className="bubble">
              <span className="dots" aria-label="Waiting for reply">
                <i />
                <i />
                <i />
              </span>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </main>

      <footer className="composer">
        <textarea
          ref={inputRef}
          data-testid="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
          rows={1}
          autoFocus
        />
        <button
          type="button"
          data-testid="send-button"
          onClick={send}
          disabled={sending || !input.trim() || tooLong}
        >
          {sending ? "Sending…" : "Send"}
        </button>
      </footer>
      {tooLong && <p className="limit">Messages are limited to 4000 characters.</p>}
    </div>
  );
}
