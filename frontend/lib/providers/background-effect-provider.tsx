"use client";

/**
 * Background effect provider for managing raining currency animation
 */

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

interface BackgroundEffectProviderProps {
  children: ReactNode;
}

interface BackgroundEffectProviderState {
  isEnabled: boolean;
  toggleEffect: () => void;
}

const BackgroundEffectContext = createContext<BackgroundEffectProviderState | undefined>(undefined);

const STORAGE_KEY = "poe-background-effect";

/**
 * Background effect provider component
 */
export function BackgroundEffectProvider({ children }: BackgroundEffectProviderProps) {
  // Always start with false to match SSR
  const [isEnabled, setIsEnabled] = useState<boolean>(false);
  const [mounted, setMounted] = useState<boolean>(false);

  // After hydration, read from localStorage
  useEffect(() => {
    setMounted(true);
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "true") {
      setIsEnabled(true);
    }
  }, []);

  // Save to localStorage when changed (but only after mount)
  useEffect(() => {
    if (mounted) {
      localStorage.setItem(STORAGE_KEY, String(isEnabled));
    }
  }, [isEnabled, mounted]);

  const toggleEffect = () => {
    setIsEnabled((prev) => !prev);
  };

  const value: BackgroundEffectProviderState = {
    isEnabled,
    toggleEffect,
  };

  return (
    <BackgroundEffectContext.Provider value={value}>
      {children}
    </BackgroundEffectContext.Provider>
  );
}

/**
 * Hook to use background effect context
 */
export function useBackgroundEffect() {
  const context = useContext(BackgroundEffectContext);

  if (context === undefined) {
    throw new Error("useBackgroundEffect must be used within a BackgroundEffectProvider");
  }

  return context;
}

