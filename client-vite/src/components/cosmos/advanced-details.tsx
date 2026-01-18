import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";
import type { Span, Cluster, Baseline } from "@/types/cosmos";
import { shorten, formatBaselineName } from "@/hooks/use-cosmos";
import { cn } from "@/lib/utils";

interface SpanItemProps {
  span: Span;
}

function SpanItem({ span }: SpanItemProps) {
  return (
    <div
      className={cn(
        "rounded-md p-2 border text-sm",
        span.selected
          ? "border-primary bg-primary/5"
          : "border-border bg-muted/30",
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-xs text-muted-foreground">
          #{span.id}
        </span>
        <span className="text-xs text-muted-foreground">
          {span.token_count}t
        </span>
        {span.weight !== undefined && (
          <span className="text-xs text-muted-foreground">
            w:{span.weight.toFixed(2)}
          </span>
        )}
        {span.selected && (
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            kept
          </Badge>
        )}
        {span.must_keep && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            must-keep
          </Badge>
        )}
        {span.is_heading && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            heading
          </Badge>
        )}
      </div>
      <div className="font-mono text-xs text-foreground leading-relaxed line-clamp-2">
        {span.text}
      </div>
    </div>
  );
}

interface ClusterItemProps {
  cluster: Cluster;
  spans: Span[];
}

function ClusterItem({ cluster, spans }: ClusterItemProps) {
  return (
    <div className="p-2 rounded-md bg-muted/30 border border-dashed text-sm">
      <div className="text-xs text-muted-foreground mb-1.5">
        Cluster {cluster.cluster}
      </div>
      <div className="flex flex-wrap gap-1">
        {cluster.spans.map((sid) => {
          const spanText = spans.find((s) => s.id === sid)?.text || "";
          return (
            <span
              key={sid}
              className="inline-block px-2 py-0.5 rounded text-xs bg-muted border"
            >
              #{sid} {shorten(spanText, 30)}
            </span>
          );
        })}
      </div>
    </div>
  );
}

interface BaselineItemProps {
  baseline: Baseline;
}

function BaselineItem({ baseline }: BaselineItemProps) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-muted-foreground">
        {formatBaselineName(baseline.name)}
      </span>
      <span className="font-mono text-xs">
        {baseline.metrics.compressed_tokens}t ·{" "}
        {baseline.metrics.savings_percent}%
      </span>
    </div>
  );
}

interface AdvancedDetailsProps {
  spans: Span[];
  clusters: Cluster[];
  baselines: Baseline[];
}

export function AdvancedDetails({
  spans,
  clusters,
  baselines,
}: AdvancedDetailsProps) {
  const [isOpen, setIsOpen] = useState(false);

  const hasData =
    spans.length > 0 || clusters.length > 0 || baselines.length > 0;

  if (!hasData) {
    return null;
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <Card>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-medium">
                Advanced details
              </CardTitle>
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  isOpen && "rotate-180",
                )}
              />
            </div>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            {/* Spans */}
            {spans.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <h4 className="text-sm font-medium">Spans</h4>
                  <Badge variant="secondary" className="font-normal text-xs">
                    {spans.filter((s) => s.selected).length}/{spans.length} kept
                  </Badge>
                </div>
                <div className="max-h-[240px] overflow-y-auto space-y-1.5 pr-1">
                  {spans.map((span) => (
                    <SpanItem key={span.id} span={span} />
                  ))}
                </div>
              </div>
            )}

            {/* Clusters */}
            {clusters.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <h4 className="text-sm font-medium">Clusters</h4>
                  <Badge variant="secondary" className="font-normal text-xs">
                    {clusters.length} groups
                  </Badge>
                </div>
                <div className="space-y-1.5">
                  {clusters.map((cluster) => (
                    <ClusterItem
                      key={cluster.cluster}
                      cluster={cluster}
                      spans={spans}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Baselines */}
            {baselines.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Baselines</h4>
                <div className="divide-y">
                  {baselines.map((baseline) => (
                    <BaselineItem key={baseline.name} baseline={baseline} />
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
