import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Re-export all utility functions
export * from "./utils/format";
export * from "./utils/profit-calculations";
export * from "./utils/risk-calculations";
export * from "./utils/filtering";
export * from "./utils/sorting";
export * from "./utils/validation";
