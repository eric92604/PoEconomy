import Image from "next/image";
import { cn } from "@/lib/utils";
import { memo, useState, useEffect } from "react";
import { getOptimizedCurrencyIcon } from "@/lib/constants/currency-icons";

interface CurrencyIconProps {
  iconUrl?: string;
  currency: string;
  size?: "sm" | "md" | "lg";
  className?: string;
  priority?: boolean;
  lazy?: boolean;
}

const sizeClasses = {
  sm: "w-4 h-4",
  md: "w-6 h-6", 
  lg: "w-8 h-8",
};


// Icon preloading hook
function useIconPreloader(iconUrl?: string, currency?: string) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    if (!iconUrl || !currency) return;

    // Check if image is already cached
    const img = new window.Image();
    img.onload = () => setIsLoaded(true);
    img.onerror = () => setHasError(true);
    img.src = iconUrl;
  }, [iconUrl, currency]);

  return { isLoaded, hasError };
}

export const CurrencyIcon = memo(function CurrencyIcon({ 
  iconUrl, 
  currency, 
  size = "md", 
  className,
  priority = false,
  lazy = true
}: CurrencyIconProps) {
  const finalIconUrl = iconUrl || getOptimizedCurrencyIcon(currency);
  const { hasError } = useIconPreloader(finalIconUrl, currency);
  const [isInView, setIsInView] = useState(!lazy);

  // Intersection Observer for lazy loading
  useEffect(() => {
    if (!lazy) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: '50px' }
    );

    const element = document.getElementById(`currency-icon-${currency}`);
    if (element) {
      observer.observe(element);
    }

    return () => observer.disconnect();
  }, [currency, lazy]);

  // Fallback for missing or failed icons
  if (!finalIconUrl || hasError) {
    return (
      <div 
        id={`currency-icon-${currency}`}
        className={cn(
          "flex items-center justify-center rounded-full bg-muted text-muted-foreground font-semibold",
          sizeClasses[size],
          className
        )}
      >
        {currency.charAt(0).toUpperCase()}
      </div>
    );
  }

  // Don't render until in view (for lazy loading)
  if (lazy && !isInView) {
    return (
      <div 
        id={`currency-icon-${currency}`}
        className={cn(
          "flex items-center justify-center rounded-full bg-muted text-muted-foreground font-semibold",
          sizeClasses[size],
          className
        )}
      >
        {currency.charAt(0).toUpperCase()}
      </div>
    );
  }

  return (
    <div 
      id={`currency-icon-${currency}`}
      className={cn("relative flex-shrink-0", sizeClasses[size], className)}
    >
      <Image
        src={finalIconUrl}
        alt={`${currency} icon`}
        width={size === "sm" ? 16 : size === "md" ? 24 : 32}
        height={size === "sm" ? 16 : size === "md" ? 24 : 32}
        className="rounded-sm"
        unoptimized={false} // Local AVIF icons can be optimized
        priority={priority}
        loading={lazy ? "lazy" : "eager"}
        onLoad={() => {}}
        onError={() => {}}
      />
    </div>
  );
});
