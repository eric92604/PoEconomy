import Image from "next/image";
import { cn } from "@/lib/utils";
import { memo, useState, useCallback } from "react";
import { getOptimizedCurrencyIcon } from "@/lib/constants/currency-icons";

interface CurrencyIconProps {
  iconUrl?: string;
  currency: string;
  size?: "sm" | "md" | "lg";
  className?: string;
  priority?: boolean;
}

const sizeMap = {
  sm: { class: "w-4 h-4", px: 16 },
  md: { class: "w-6 h-6", px: 24 },
  lg: { class: "w-8 h-8", px: 32 },
} as const;

export const CurrencyIcon = memo(function CurrencyIcon({ 
  iconUrl, 
  currency, 
  size = "md", 
  className,
  priority = false,
}: CurrencyIconProps) {
  const [hasError, setHasError] = useState(false);
  const finalIconUrl = iconUrl || getOptimizedCurrencyIcon(currency);
  const { class: sizeClass, px } = sizeMap[size];

  const handleError = useCallback(() => setHasError(true), []);

  // Fallback for missing or failed icons
  if (!finalIconUrl || hasError) {
    return (
      <div 
        className={cn(
          "flex items-center justify-center rounded-full bg-muted text-muted-foreground font-semibold text-xs",
          sizeClass,
          className
        )}
      >
        {currency.charAt(0).toUpperCase()}
      </div>
    );
  }

  return (
    <div className={cn("relative flex-shrink-0", sizeClass, className)}>
      <Image
        src={finalIconUrl}
        alt={`${currency} icon`}
        width={px}
        height={px}
        className="rounded-sm"
        priority={priority}
        loading={priority ? "eager" : "lazy"}
        onError={handleError}
      />
    </div>
  );
});
