import { useState, useCallback, useEffect } from "react";
import type {
  Toggles,
  CompressionResponse,
  TokenCoResult,
  Scenario,
  RawScenario,
} from "@/types/cosmos";
import { config } from "@/config";

const API_BASE = config.apiBase;

// Simple token counter (word-based approximation)
export function countTokens(text: string): number {
  const tokens = text.match(/[A-Za-z0-9']+/g) || [];
  return tokens.length;
}

export function shorten(text: string, max = 80): string {
  if (!text) return "";
  if (text.length <= max) return text;
  return `${text.slice(0, max - 3)}...`;
}

export function formatCategory(category: string): string {
  const labels: Record<string, string> = {
    rag_overload: "RAG overload",
    meeting_transcript: "Meeting transcript",
    policy_doc: "Policy doc",
  };
  if (!category) return "Scenario";
  if (labels[category]) return labels[category];
  return category.replace(/_/g, " ");
}

export function formatBaselineName(name: string): string {
  if (!name) return "Baseline";
  return name
    .split("_")
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

export function normalizeScenario(raw: RawScenario, index: number): Scenario {
  const query = raw.query || "";
  const text = raw.text || "";
  const title = query
    ? shorten(query, 56)
    : shorten(text.split("\n")[0] || "Scenario", 56);
  const summary = shorten(text.replace(/\s+/g, " ").trim(), 92);
  return {
    id: raw.id ?? index,
    category: raw.category || "scenario",
    query,
    text,
    title,
    summary,
    tokens: countTokens(text),
  };
}

export interface CosmosState {
  // Input state
  query: string;
  context: string;
  ratio: number;
  lastN: number;
  apiKey: string;

  // Toggle state
  keepEntities: boolean;
  keepHeadings: boolean;
  useSignal: boolean;
  signalBoost: number;
  noveltyBoost: number;
  paraphraseMode: "none" | "heuristic" | "llm";

  // Result state
  compressionResult: CompressionResponse | null;
  tokenCoResult: TokenCoResult | null;

  // Scenario state
  scenarios: Scenario[];
  selectedScenarioIndex: number;

  // UI state
  status: string;
  statusError: boolean;
  isLoading: boolean;
  showTokenCoOutput: boolean;
  showComparison: boolean;
}

export interface CosmosActions {
  setQuery: (query: string) => void;
  setContext: (context: string) => void;
  setRatio: (ratio: number) => void;
  setLastN: (lastN: number) => void;
  setApiKey: (apiKey: string) => void;
  setKeepEntities: (value: boolean) => void;
  setKeepHeadings: (value: boolean) => void;
  setUseSignal: (value: boolean) => void;
  setSignalBoost: (value: number) => void;
  setNoveltyBoost: (value: number) => void;
  setParaphraseMode: (mode: "none" | "heuristic" | "llm") => void;
  selectScenario: (index: number) => void;
  compress: () => Promise<void>;
  compare: () => Promise<void>;
  inputTokens: number;
}

export function useCosmos(
  fallbackScenarios: RawScenario[],
): CosmosState & CosmosActions {
  // Input state
  const [query, setQuery] = useState("");
  const [context, setContext] = useState("");
  const [ratio, setRatio] = useState(0.5);
  const [lastN, setLastN] = useState(1);
  const [apiKey, setApiKey] = useState("");

  // Toggle state
  const [keepEntities, setKeepEntities] = useState(true);
  const [keepHeadings, setKeepHeadings] = useState(true);
  const [useSignal, setUseSignal] = useState(true);
  const [signalBoost, setSignalBoost] = useState(0.65);
  const [noveltyBoost, setNoveltyBoost] = useState(0.35);
  const [paraphraseMode, setParaphraseMode] = useState<
    "none" | "heuristic" | "llm"
  >("none");

  // Result state
  const [compressionResult, setCompressionResult] =
    useState<CompressionResponse | null>(null);
  const [tokenCoResult, setTokenCoResult] = useState<TokenCoResult | null>(
    null,
  );

  // Scenario state
  const [scenarios, setScenarios] = useState<Scenario[]>(() =>
    fallbackScenarios.map(normalizeScenario),
  );
  const [selectedScenarioIndex, setSelectedScenarioIndex] = useState(0);

  // UI state
  const [status, setStatus] = useState("Waiting for input.");
  const [statusError, setStatusError] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showTokenCoOutput, setShowTokenCoOutput] = useState(false);
  const [showComparison, setShowComparison] = useState(false);

  const inputTokens = countTokens(context);

  const updateStatus = useCallback((message: string, isError = false) => {
    setStatus(message);
    setStatusError(isError);
  }, []);

  const buildToggles = useCallback((): Toggles => {
    return {
      keep_numbers_entities: keepEntities,
      keep_headings: keepHeadings,
      keep_code_blocks: true,
      keep_role_markers: true,
      use_signal_scores: useSignal,
      signal_boost: signalBoost,
      novelty_boost: noveltyBoost,
      paraphrase_mode: paraphraseMode,
    };
  }, [
    keepEntities,
    keepHeadings,
    useSignal,
    signalBoost,
    noveltyBoost,
    paraphraseMode,
  ]);

  const selectScenario = useCallback(
    (index: number) => {
      const scenario = scenarios[index];
      if (!scenario) return;
      setSelectedScenarioIndex(index);
      setQuery(scenario.query || "");
      setContext(scenario.text || "");
      // Clear previous results when switching scenarios
      setCompressionResult(null);
      setTokenCoResult(null);
      setShowTokenCoOutput(false);
      setShowComparison(false);
      updateStatus(`Loaded scenario: ${scenario.title}`);
    },
    [scenarios, updateStatus],
  );

  const compress = useCallback(async () => {
    updateStatus("Compressing with COSMOS...");
    setShowTokenCoOutput(false);
    setShowComparison(false);
    setIsLoading(true);
    setTokenCoResult(null);

    const toggles = buildToggles();
    const payload = {
      text: context,
      query,
      target_ratio: 1 - ratio,
      keep_last_n: lastN,
      run_baselines: true,
      toggles,
    };

    const url = `${API_BASE}/compress`;
    console.log("[COSMOS] Compressing to:", url);
    console.log("[COSMOS] Payload:", payload);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error("[COSMOS] API error:", res.status, errorText);
        updateStatus(
          `API error: ${res.status} - ${errorText.slice(0, 100)}`,
          true,
        );
        return;
      }

      const data: CompressionResponse = await res.json();
      console.log("[COSMOS] Response:", data);
      setCompressionResult(data);
      updateStatus(
        `Done. Saved ${data.metrics?.savings_percent ?? "n/a"}% tokens.`,
      );
    } catch (err) {
      console.error("[COSMOS] Fetch error:", err);
      const message = err instanceof Error ? err.message : "Unknown error";
      updateStatus(`Compression failed: ${message}`, true);
    } finally {
      setIsLoading(false);
    }
  }, [context, query, ratio, lastN, buildToggles, updateStatus]);

  const compare = useCallback(async () => {
    updateStatus("Running comparison...");
    setShowTokenCoOutput(false);
    setShowComparison(false);
    setIsLoading(true);

    const toggles = buildToggles();
    const payload = {
      text: context,
      query,
      target_ratio: 1 - ratio,
      aggressiveness: ratio,
      api_key: apiKey || null,
      toggles,
    };

    const url = `${API_BASE}/compare`;
    console.log("[COSMOS] Comparing to:", url);
    console.log("[COSMOS] Payload:", payload);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error("[COSMOS] API error:", res.status, errorText);
        updateStatus(
          `API error: ${res.status} - ${errorText.slice(0, 100)}`,
          true,
        );
        return;
      }

      const data = await res.json();
      console.log("[COSMOS] Response:", data);
      const cosmos = data.cosmos as CompressionResponse;
      const tokenc = data.tokenc?.available
        ? (data.tokenc as TokenCoResult)
        : null;

      setCompressionResult(cosmos);
      setTokenCoResult(tokenc);

      if (tokenc) {
        setShowTokenCoOutput(true);
        setShowComparison(true);
      }

      updateStatus(
        tokenc ? "Compared vs TokenCo" : "COSMOS done; TokenCo unavailable",
      );
    } catch (err) {
      console.error("[COSMOS] Fetch error:", err);
      const message = err instanceof Error ? err.message : "Unknown error";
      updateStatus(`Comparison failed: ${message}`, true);
    } finally {
      setIsLoading(false);
    }
  }, [context, query, ratio, apiKey, buildToggles, updateStatus]);

  // Load scenarios from API on mount
  useEffect(() => {
    async function loadScenarios() {
      const url = `${API_BASE}/examples`;
      console.log("[COSMOS] Loading examples from:", url);
      try {
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          console.log("[COSMOS] Examples loaded:", data);
          if (data.examples?.length) {
            const normalized = data.examples.map(normalizeScenario);
            setScenarios(normalized);
          }
        } else {
          console.warn("[COSMOS] Failed to load examples:", res.status);
        }
      } catch (err) {
        console.warn("[COSMOS] Failed to load examples:", err);
      }
    }
    loadScenarios();
  }, []);

  // Select first scenario on mount if context is empty
  useEffect(() => {
    if (!context && scenarios.length > 0) {
      selectScenario(0);
    }
  }, [scenarios]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    // State
    query,
    context,
    ratio,
    lastN,
    apiKey,
    keepEntities,
    keepHeadings,
    useSignal,
    signalBoost,
    noveltyBoost,
    paraphraseMode,
    compressionResult,
    tokenCoResult,
    scenarios,
    selectedScenarioIndex,
    status,
    statusError,
    isLoading,
    showTokenCoOutput,
    showComparison,
    inputTokens,

    // Actions
    setQuery,
    setContext,
    setRatio,
    setLastN,
    setApiKey,
    setKeepEntities,
    setKeepHeadings,
    setUseSignal,
    setSignalBoost,
    setNoveltyBoost,
    setParaphraseMode,
    selectScenario,
    compress,
    compare,
  };
}
