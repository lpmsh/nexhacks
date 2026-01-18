import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Span, Metrics, Toggles } from "@/types/cosmos";
import { shorten } from "@/hooks/use-cosmos";

interface ProcessStepProps {
  title: string;
  description: string;
  children: React.ReactNode;
  delay?: string;
}

function ProcessStep({ title, description, children }: ProcessStepProps) {
  return (
    <Card className="bg-white border-border ">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">{title}</CardTitle>
        <p className="text-sm text-muted-foreground">{description}</p>
      </CardHeader>
      <CardContent className="pt-0">{children}</CardContent>
    </Card>
  );
}

interface SignalItemProps {
  weight: number | undefined;
  tokenCount: number | undefined;
  text: string;
  tags: string[];
}

function SignalItem({ weight, tokenCount, text, tags }: SignalItemProps) {
  const tagText = tags.length ? `(${tags.join(", ")}) ` : "";
  return (
    <div className="bg-muted/50 border border-border  p-2.5">
      <div className="flex justify-between gap-1.5 text-[11px] text-muted-foreground uppercase tracking-wide">
        <span>weight {weight?.toFixed?.(2) ?? "-"}</span>
        <span>{tokenCount ?? "-"} tokens</span>
      </div>
      <div className="mt-1.5 font-mono text-xs text-foreground leading-relaxed">
        {tagText}
        {shorten(text, 120)}
      </div>
    </div>
  );
}

interface ProcessPipelineProps {
  spans: Span[];
  metrics: Metrics | null;
  toggles: Toggles | null;
  budget: number | null;
  spansKept: number | null;
  spansTotal: number | null;
}

export function ProcessPipeline({
  spans,
  metrics,
  toggles,
  budget,
  spansKept,
  spansTotal,
}: ProcessPipelineProps) {
  const total = spans?.length || 0;
  const mustKeep = spans?.filter((s) => s.must_keep).length || 0;
  const headings = spans?.filter((s) => s.is_heading).length || 0;

  // Get top 4 spans by weight for signal list
  const topSignals = [...(spans || [])]
    .filter((span) => span.weight !== undefined)
    .sort((a, b) => (b.weight || 0) - (a.weight || 0))
    .slice(0, 4);

  return (
    <section className="mt-8">
      <div className="flex justify-between gap-4 items-baseline flex-wrap mb-4">
        <h2 className="text-2xl font-serif font-semibold">
          Compression pipeline
        </h2>
        <p className="text-muted-foreground max-w-xl">
          Clear, compact view of the model steps and what COSMOS keeps.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        <ProcessStep
          title="1. Chunk + guardrails"
          description="Split context into spans, keep headings, enforce code/role constraints."
        >
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">{total} spans</Badge>
            <Badge variant="secondary">must-keep {mustKeep}</Badge>
            <Badge variant="secondary">headings {headings}</Badge>
          </div>
        </ProcessStep>

        <ProcessStep
          title="2. Signal weighting"
          description="Representation-drop and entity boosts elevate the most critical spans."
        >
          <div className="grid gap-2 max-h-[200px] overflow-y-auto">
            {topSignals.length > 0 ? (
              topSignals.map((span) => {
                const tags: string[] = [];
                if (span.must_keep) tags.push("must-keep");
                if (span.is_heading) tags.push("heading");
                if (span.is_question) tags.push("query");
                return (
                  <SignalItem
                    key={span.id}
                    weight={span.weight}
                    tokenCount={span.token_count}
                    text={span.text}
                    tags={tags}
                  />
                );
              })
            ) : (
              <div className="text-sm text-muted-foreground">
                Run compression to see top signals.
              </div>
            )}
          </div>
        </ProcessStep>

        <ProcessStep
          title="3. Facility-location selection"
          description="Greedy selection maximizes coverage under a strict token budget."
        >
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">Budget: {budget ?? "-"}</Badge>
            <Badge variant="secondary">
              Coverage: {metrics?.coverage_score ?? "-"}
            </Badge>
            <Badge variant="secondary">
              Kept:{" "}
              {spansKept !== null && spansTotal !== null
                ? `${spansKept}/${spansTotal}`
                : "-"}
            </Badge>
          </div>
        </ProcessStep>

        <ProcessStep
          title="4. Paraphrase squeeze"
          description="Optional constrained paraphrase squeezes extra tokens without losing structure."
        >
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              Mode: {toggles?.paraphrase_mode ?? "none"}
            </Badge>
            <Badge variant="secondary">
              Output: {metrics?.compressed_tokens ?? "-"}
            </Badge>
          </div>
        </ProcessStep>
      </div>
    </section>
  );
}
