import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useCosmos } from "@/hooks/use-cosmos";
import { FALLBACK_SCENARIOS } from "@/types/cosmos";
import { HeroMetrics } from "./hero-metrics";
import { ScenarioDeck } from "./scenario-deck";
import { ExperimentControls } from "./experiment-controls";
import { InputOutputPane } from "./input-output-pane";
import { ImpactChart } from "./impact-chart";
import { AdvancedDetails } from "./advanced-details";
import { cn } from "@/lib/utils";

export function CosmosPage() {
  const cosmos = useCosmos(FALLBACK_SCENARIOS);

  const toggles = {
    keep_numbers_entities: cosmos.keepEntities,
    keep_headings: cosmos.keepHeadings,
    keep_code_blocks: true,
    keep_role_markers: true,
    use_signal_scores: cosmos.useSignal,
    signal_boost: cosmos.signalBoost,
    novelty_boost: cosmos.noveltyBoost,
    paraphrase_mode: cosmos.paraphraseMode,
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto p-8">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-2 mb-2">
            <Badge variant="secondary" className="text-xs font-normal">
              Track 1
            </Badge>
            <span className="text-sm text-muted-foreground">
              Alternative Compression Model
            </span>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight mb-2">COSMOS</h1>
          <p className="text-muted-foreground max-w-2xl">
            Facility-location compression with representation-drop signals,
            guardrails, and query awareness.
          </p>
        </header>

        {/* Metrics Overview */}
        <section className="mb-8">
          <HeroMetrics
            metrics={cosmos.compressionResult?.metrics ?? null}
            budget={cosmos.compressionResult?.budget ?? null}
            spanCounts={cosmos.compressionResult?.span_counts ?? null}
          />
        </section>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
          {/* Left Column - Main Workspace */}
          <div className="space-y-6">
            {/* Input/Output */}
            <InputOutputPane
              context={cosmos.context}
              onContextChange={cosmos.setContext}
              query={cosmos.query}
              onQueryChange={cosmos.setQuery}
              inputTokens={cosmos.inputTokens}
              compressedText={cosmos.compressionResult?.compressed_text ?? null}
              metrics={cosmos.compressionResult?.metrics ?? null}
              budget={cosmos.compressionResult?.budget ?? null}
              spansKept={
                cosmos.compressionResult?.span_counts?.selected ?? null
              }
              spansTotal={cosmos.compressionResult?.span_counts?.total ?? null}
              showTokenCoOutput={cosmos.showTokenCoOutput}
              showComparison={cosmos.showComparison}
              tokenCoResult={cosmos.tokenCoResult}
            />

            {/* Impact Chart */}
            <ImpactChart
              inputTokens={
                cosmos.compressionResult?.input_tokens ?? cosmos.inputTokens
              }
              cosmosTokens={
                cosmos.compressionResult?.metrics?.compressed_tokens ?? null
              }
              tokencTokens={
                cosmos.tokenCoResult?.available
                  ? (cosmos.tokenCoResult.output_tokens ??
                    cosmos.tokenCoResult.metrics?.compressed_tokens ??
                    null)
                  : null
              }
              baselines={cosmos.compressionResult?.baselines ?? []}
            />

            {/* Advanced Details */}
            <AdvancedDetails
              spans={cosmos.compressionResult?.spans ?? []}
              clusters={cosmos.compressionResult?.clusters ?? []}
              baselines={cosmos.compressionResult?.baselines ?? []}
            />
          </div>

          {/* Right Column - Settings */}
          <div className="space-y-6">
            {/* Action Buttons */}
            <div className="space-y-3">
              <div className="flex flex-col gap-2">
                <Button
                  onClick={cosmos.compress}
                  disabled={cosmos.isLoading}
                  className="w-full bg-primary hover:bg-primary/90"
                >
                  {cosmos.isLoading ? "Compressing..." : "Compress"}
                </Button>
                <Button
                  variant="outline"
                  onClick={cosmos.compare}
                  disabled={cosmos.isLoading}
                  className="w-full"
                >
                  Compare vs TokenCo
                </Button>
              </div>
              <p
                className={cn(
                  "text-xs font-mono text-center",
                  cosmos.statusError
                    ? "text-destructive"
                    : "text-muted-foreground",
                )}
              >
                {cosmos.status}
              </p>
            </div>

            {/* Scenario Selection */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base font-medium">
                  Scenarios
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <ScenarioDeck
                  scenarios={cosmos.scenarios}
                  selectedIndex={cosmos.selectedScenarioIndex}
                  onSelect={cosmos.selectScenario}
                />
              </CardContent>
            </Card>

            {/* Controls */}
            <ExperimentControls
              apiKey={cosmos.apiKey}
              onApiKeyChange={cosmos.setApiKey}
              ratio={cosmos.ratio}
              onRatioChange={cosmos.setRatio}
              lastN={cosmos.lastN}
              onLastNChange={cosmos.setLastN}
              keepEntities={cosmos.keepEntities}
              onKeepEntitiesChange={cosmos.setKeepEntities}
              keepHeadings={cosmos.keepHeadings}
              onKeepHeadingsChange={cosmos.setKeepHeadings}
              useSignal={cosmos.useSignal}
              onUseSignalChange={cosmos.setUseSignal}
              signalBoost={cosmos.signalBoost}
              onSignalBoostChange={cosmos.setSignalBoost}
              noveltyBoost={cosmos.noveltyBoost}
              onNoveltyBoostChange={cosmos.setNoveltyBoost}
              paraphraseMode={cosmos.paraphraseMode}
              onParaphraseModeChange={cosmos.setParaphraseMode}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
