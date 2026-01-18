import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import type { Metrics, TokenCoResult } from "@/types/cosmos";

interface InputOutputPaneProps {
  context: string;
  onContextChange: (value: string) => void;
  query: string;
  onQueryChange: (value: string) => void;
  inputTokens: number;
  compressedText: string | null;
  metrics: Metrics | null;
  budget: number | null;
  spansKept: number | null;
  spansTotal: number | null;
  showTokenCoOutput: boolean;
  showComparison: boolean;
  tokenCoResult: TokenCoResult | null;
}

function extractTokencTokens(tokenc: TokenCoResult | null): number | null {
  if (!tokenc) return null;
  return tokenc.output_tokens ?? tokenc.metrics?.compressed_tokens ?? null;
}

function extractTokencSavings(tokenc: TokenCoResult | null): number | null {
  if (!tokenc) return null;
  const savings =
    tokenc.compression_percentage ?? tokenc.metrics?.savings_percent;
  return savings !== undefined && savings !== null ? Number(savings) : null;
}

export function InputOutputPane({
  context,
  onContextChange,
  query,
  onQueryChange,
  inputTokens,
  compressedText,
  metrics,
  budget,
  spansKept,
  spansTotal,
  showTokenCoOutput,
  showComparison,
  tokenCoResult,
}: InputOutputPaneProps) {
  const tokencSavings = extractTokencSavings(tokenCoResult);
  const cosmosSavings = metrics?.savings_percent;

  let comparisonText = "";
  if (showComparison && tokencSavings !== null && cosmosSavings !== undefined) {
    const delta = cosmosSavings - tokencSavings;
    if (delta >= 0) {
      comparisonText = `COSMOS saves ${delta.toFixed(1)}% more than TokenCo`;
    } else {
      comparisonText = `TokenCo saves ${Math.abs(delta).toFixed(1)}% more than COSMOS`;
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Input */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-medium">Input</CardTitle>
            <Badge variant="secondary" className="font-normal">
              {inputTokens} tokens
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            placeholder="Paste your long context, retrieval set, or transcript..."
            rows={14}
            value={context}
            onChange={(e) => onContextChange(e.target.value)}
            className="font-mono text-sm resize-y min-h-[280px] max-h-[400px] overflow-y-auto"
          />
          <div className="space-y-1.5">
            <Label htmlFor="query" className="text-sm">
              Query (optional)
            </Label>
            <Input
              id="query"
              type="text"
              placeholder="What should the assistant answer?"
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Output */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-medium">Output</CardTitle>
            {metrics?.compressed_tokens !== undefined && (
              <Badge variant="secondary" className="font-normal">
                {metrics.compressed_tokens} tokens
                {metrics.savings_percent !== undefined &&
                  ` · ${metrics.savings_percent}% saved`}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Comparison Banner */}
          {showComparison && comparisonText && (
            <div className="text-sm px-3 py-2  bg-muted border text-muted-foreground">
              {comparisonText}
            </div>
          )}

          {/* COSMOS Output */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">COSMOS</span>
              {metrics?.coverage_score !== undefined && (
                <span className="text-xs text-muted-foreground">
                  Coverage: {metrics.coverage_score}
                </span>
              )}
            </div>
            <div
              className="min-h-[200px] max-h-[300px] overflow-y-auto border bg-muted/30 p-3 font-mono text-sm whitespace-pre-wrap"
              aria-live="polite"
            >
              {compressedText || (
                <span className="text-muted-foreground">
                  Run compression to see output.
                </span>
              )}
            </div>
          </div>

          {/* TokenCo Output */}
          {showTokenCoOutput && tokenCoResult && (
            <div className="space-y-2 pt-2 border-t">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">TokenCo</span>
                <span className="text-xs text-muted-foreground">
                  {extractTokencTokens(tokenCoResult) ?? "-"} tokens
                  {tokencSavings !== null && ` · ${tokencSavings.toFixed(1)}%`}
                </span>
              </div>
              <div className="min-h-[120px] max-h-[200px] overflow-y-auto border bg-muted/30 p-3 font-mono text-sm whitespace-pre-wrap text-muted-foreground">
                {tokenCoResult.text || "No output received."}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
