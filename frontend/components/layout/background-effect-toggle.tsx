"use client";

/**
 * Toggle button for enabling/disabling the raining currency background effect
 */

import { Sparkles } from "lucide-react";
import { useBackgroundEffect } from "@/lib/providers/background-effect-provider";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export function BackgroundEffectToggle() {
  const { isEnabled, toggleEffect } = useBackgroundEffect();

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="icon"
            onClick={toggleEffect}
            className={cn(
              "transition-all",
              isEnabled && "bg-primary/10 border-primary/50"
            )}
          >
            <Sparkles 
              className={cn(
                "h-[1.2rem] w-[1.2rem] transition-all",
                isEnabled && "text-primary"
              )} 
            />
            <span className="sr-only">Toggle background effect</span>
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Make it rain!</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

