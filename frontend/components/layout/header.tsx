"use client";

/**
 * Header component with navigation and theme toggle
 */

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./theme-toggle";
import { BackgroundEffectToggle } from "./background-effect-toggle";
import { APP_NAME } from "@/lib/constants/config";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Dashboard", href: "/dashboard" },
  { name: "Investments", href: "/investments" },
  { name: "Live Prices", href: "/prices" },
  { name: "League History", href: "/league-history" },
];

export function Header() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[var(--poe-gold)]/20 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex h-14 items-center">
        {/* Logo */}
        <Link href="/" className="mr-6 flex items-center space-x-2 group">
          <Image
            src="/images/mirror-of-kalandra.png"
            alt="PoEconomy"
            width={28}
            height={28}
            className="group-hover:drop-shadow-[0_0_8px_rgba(201,169,97,0.5)] transition-all duration-300"
          />
          <span className="font-bold text-xl text-poe-gold">{APP_NAME}</span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center space-x-6 text-sm font-medium flex-1">
          {navigation.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "transition-colors hover:text-poe-gold",
                pathname === item.href 
                  ? "text-poe-gold" 
                  : "text-foreground/60"
              )}
            >
              {item.name}
            </Link>
          ))}
        </nav>

        {/* Controls */}
        <div className="flex items-center justify-end space-x-2">
          <BackgroundEffectToggle />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

