import { cn } from "@/lib/utils";
import type { Scenario } from "@/types/cosmos";
import { formatCategory } from "@/hooks/use-cosmos";

interface ScenarioDeckProps {
  scenarios: Scenario[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}

export function ScenarioDeck({
  scenarios,
  selectedIndex,
  onSelect,
}: ScenarioDeckProps) {
  return (
    <div className="space-y-1">
      {scenarios.map((scenario, index) => (
        <button
          key={scenario.id}
          onClick={() => onSelect(index)}
          className={cn(
            "w-full text-left px-3 py-2 rounded-md text-sm transition-colors",
            index === selectedIndex
              ? "bg-primary text-primary-foreground"
              : "hover:bg-muted",
          )}
        >
          <div className="font-medium truncate">{scenario.title}</div>
          <div
            className={cn(
              "text-xs truncate",
              index === selectedIndex
                ? "text-primary-foreground/70"
                : "text-muted-foreground",
            )}
          >
            {formatCategory(scenario.category)} · {scenario.tokens} tokens
          </div>
        </button>
      ))}
    </div>
  );
}
