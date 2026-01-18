import type { Metrics, SpanCounts } from "@/types/cosmos";

interface MetricItemProps {
  label: string;
  value: string | number;
}

function MetricItem({ label, value }: MetricItemProps) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-2xl font-semibold tabular-nums">{value}</span>
    </div>
  );
}

interface HeroMetricsProps {
  metrics: Metrics | null;
  budget: number | null;
  spanCounts: SpanCounts | null;
}

export function HeroMetrics({ metrics, budget, spanCounts }: HeroMetricsProps) {
  const savings =
    metrics?.savings_percent !== undefined
      ? `${metrics.savings_percent}%`
      : "-";
  const coverage =
    metrics?.coverage_score !== undefined
      ? String(metrics.coverage_score)
      : "-";
  const budgetDisplay = budget !== null ? String(budget) : "-";
  const spans =
    spanCounts?.selected !== undefined && spanCounts?.total !== undefined
      ? `${spanCounts.selected}/${spanCounts.total}`
      : "-";

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 p-4 bg-neutral-100 border">
      <MetricItem label="Tokens saved" value={savings} />
      <MetricItem label="Coverage" value={coverage} />
      <MetricItem label="Budget" value={budgetDisplay} />
      <MetricItem label="Spans kept" value={spans} />
    </div>
  );
}
