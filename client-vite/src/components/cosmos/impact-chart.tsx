import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Baseline } from "@/types/cosmos";
import { formatBaselineName } from "@/hooks/use-cosmos";
import { cn } from "@/lib/utils";

interface ImpactItem {
  label: string;
  tokens: number;
  kind: "input" | "cosmos" | "tokenc" | "baseline";
}

interface ImpactChartProps {
  inputTokens: number | null;
  cosmosTokens: number | null;
  tokencTokens: number | null;
  baselines: Baseline[];
}

function ImpactRow({
  item,
  maxTokens,
  inputTokens,
}: {
  item: ImpactItem;
  maxTokens: number;
  inputTokens: number;
}) {
  const width = Math.max(4, (item.tokens / maxTokens) * 100);
  const savings =
    item.kind === "input"
      ? null
      : Math.max(0, (1 - item.tokens / inputTokens) * 100);

  return (
    <div className="flex items-center gap-3 text-sm">
      <div className="w-20 shrink-0 text-muted-foreground">{item.label}</div>
      <div className="flex-1 h-2 bg-muted  overflow-hidden">
        <div
          className={cn(
            "h-full  transition-all",
            item.kind === "cosmos" && "bg-primary",
            item.kind === "tokenc" && "bg-emerald-500",
            item.kind === "input" && "bg-muted-foreground/30",
            item.kind === "baseline" && "bg-muted-foreground/50",
          )}
          style={{ width: `${width}%` }}
        />
      </div>
      <div className="w-24 shrink-0 text-right font-mono text-xs text-muted-foreground">
        {item.tokens}
        {savings !== null && (
          <span className="ml-1 text-muted-foreground/70">
            ({savings.toFixed(0)}%)
          </span>
        )}
      </div>
    </div>
  );
}

export function ImpactChart({
  inputTokens,
  cosmosTokens,
  tokencTokens,
  baselines,
}: ImpactChartProps) {
  const hasData = inputTokens !== null && cosmosTokens !== null;

  if (!hasData) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-medium">Token impact</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Run compression to see results.
          </p>
        </CardContent>
      </Card>
    );
  }

  const items: ImpactItem[] = [];

  items.push({
    label: "Input",
    tokens: inputTokens,
    kind: "input",
  });

  items.push({
    label: "COSMOS",
    tokens: cosmosTokens,
    kind: "cosmos",
  });

  if (tokencTokens !== null) {
    items.push({
      label: "TokenCo",
      tokens: tokencTokens,
      kind: "tokenc",
    });
  }

  baselines.forEach((baseline) => {
    if (!baseline?.metrics?.compressed_tokens) return;
    items.push({
      label: formatBaselineName(baseline.name),
      tokens: baseline.metrics.compressed_tokens,
      kind: "baseline",
    });
  });

  const maxTokens = Math.max(...items.map((item) => item.tokens || 0), 1);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-medium">Token impact</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {items.map((item) => (
            <ImpactRow
              key={item.label}
              item={item}
              maxTokens={maxTokens}
              inputTokens={inputTokens}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
