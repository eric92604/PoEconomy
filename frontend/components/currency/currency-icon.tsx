import Image from "next/image";
import { cn } from "@/lib/utils";

interface CurrencyIconProps {
  iconUrl?: string;
  currency: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizeClasses = {
  sm: "w-4 h-4",
  md: "w-6 h-6", 
  lg: "w-8 h-8",
};

export function CurrencyIcon({ 
  iconUrl, 
  currency, 
  size = "md", 
  className 
}: CurrencyIconProps) {
  if (!iconUrl) {
    // Fallback to a generic currency icon or just the first letter
    return (
      <div 
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
    <div className={cn("relative flex-shrink-0", sizeClasses[size], className)}>
      <Image
        src={iconUrl}
        alt={`${currency} icon`}
        width={size === "sm" ? 16 : size === "md" ? 24 : 32}
        height={size === "sm" ? 16 : size === "md" ? 24 : 32}
        className="rounded-sm"
        unoptimized // PoE CDN images don't need optimization
      />
    </div>
  );
}
