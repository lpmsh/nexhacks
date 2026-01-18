import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown, Eye, EyeOff } from "lucide-react";

interface ExperimentControlsProps {
  apiKey: string;
  onApiKeyChange: (value: string) => void;
  ratio: number;
  onRatioChange: (value: number) => void;
  lastN: number;
  onLastNChange: (value: number) => void;
  keepEntities: boolean;
  onKeepEntitiesChange: (value: boolean) => void;
  keepHeadings: boolean;
  onKeepHeadingsChange: (value: boolean) => void;
  useSignal: boolean;
  onUseSignalChange: (value: boolean) => void;
  signalBoost: number;
  onSignalBoostChange: (value: number) => void;
  noveltyBoost: number;
  onNoveltyBoostChange: (value: number) => void;
  paraphraseMode: "none" | "heuristic" | "llm";
  onParaphraseModeChange: (value: "none" | "heuristic" | "llm") => void;
}

export function ExperimentControls({
  apiKey,
  onApiKeyChange,
  ratio,
  onRatioChange,
  lastN,
  onLastNChange,
  keepEntities,
  onKeepEntitiesChange,
  keepHeadings,
  onKeepHeadingsChange,
  useSignal,
  onUseSignalChange,
  signalBoost,
  onSignalBoostChange,
  noveltyBoost,
  onNoveltyBoostChange,
  paraphraseMode,
  onParaphraseModeChange,
}: ExperimentControlsProps) {
  const [showApiKey, setShowApiKey] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-medium">Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* API Key - at top for visibility */}
        <div className="space-y-2">
          <Label htmlFor="apiKey" className="text-sm">
            TokenCo API key
          </Label>
          <div className="flex gap-1">
            <Input
              id="apiKey"
              type={showApiKey ? "text" : "password"}
              placeholder="ttc_sk_..."
              value={apiKey}
              onChange={(e) => onApiKeyChange(e.target.value)}
              className="h-8 flex-1"
            />
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0"
              onClick={() => setShowApiKey(!showApiKey)}
            >
              {showApiKey ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Required for TokenCo comparison
          </p>
        </div>

        {/* Compression Ratio */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Compression</Label>
            <span className="text-xs text-muted-foreground tabular-nums">
              {ratio.toFixed(2)}
            </span>
          </div>
          <Slider
            min={0.2}
            max={0.9}
            step={0.05}
            value={[ratio]}
            onValueChange={([value]) => onRatioChange(value ?? ratio)}
          />
          <p className="text-xs text-muted-foreground">
            Higher = more aggressive compression
          </p>
        </div>

        {/* Guardrails */}
        <div className="space-y-2">
          <Label className="text-sm">Guardrails</Label>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Checkbox
                id="keepEntities"
                checked={keepEntities}
                onCheckedChange={(checked) =>
                  onKeepEntitiesChange(checked === true)
                }
              />
              <Label
                htmlFor="keepEntities"
                className="text-sm font-normal cursor-pointer"
              >
                Protect entities/numbers
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="keepHeadings"
                checked={keepHeadings}
                onCheckedChange={(checked) =>
                  onKeepHeadingsChange(checked === true)
                }
              />
              <Label
                htmlFor="keepHeadings"
                className="text-sm font-normal cursor-pointer"
              >
                Keep headings
              </Label>
            </div>
          </div>
        </div>

        {/* Keep Last N */}
        <div className="space-y-2">
          <Label htmlFor="lastN" className="text-sm">
            Keep last N spans
          </Label>
          <Input
            id="lastN"
            type="number"
            min={0}
            max={3}
            value={lastN}
            onChange={(e) => onLastNChange(Number(e.target.value))}
            className="h-8"
          />
        </div>

        {/* Advanced Settings */}
        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-between px-0 hover:bg-transparent"
            >
              <span className="text-sm text-muted-foreground">Advanced</span>
              <ChevronDown
                className={`h-4 w-4 text-muted-foreground transition-transform ${advancedOpen ? "rotate-180" : ""}`}
              />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-4 pt-2">
            {/* Signal Scores */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Switch
                  id="useSignal"
                  checked={useSignal}
                  onCheckedChange={onUseSignalChange}
                />
                <Label htmlFor="useSignal" className="text-sm cursor-pointer">
                  Use signal scores
                </Label>
              </div>
            </div>

            {/* Signal Boost */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Signal boost</Label>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {signalBoost.toFixed(2)}
                </span>
              </div>
              <Slider
                min={0}
                max={1}
                step={0.05}
                value={[signalBoost]}
                onValueChange={([value]) =>
                  onSignalBoostChange(value ?? signalBoost)
                }
                disabled={!useSignal}
              />
            </div>

            {/* Novelty Boost */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Novelty boost</Label>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {noveltyBoost.toFixed(2)}
                </span>
              </div>
              <Slider
                min={0}
                max={1}
                step={0.05}
                value={[noveltyBoost]}
                onValueChange={([value]) =>
                  onNoveltyBoostChange(value ?? noveltyBoost)
                }
              />
            </div>

            {/* Paraphrase Mode */}
            <div className="space-y-2">
              <Label className="text-sm">Paraphrase mode</Label>
              <Select
                value={paraphraseMode}
                onValueChange={(value) =>
                  onParaphraseModeChange(value as "none" | "heuristic" | "llm")
                }
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="heuristic">Heuristic</SelectItem>
                  <SelectItem value="llm">LLM</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
}
