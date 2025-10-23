/**
 * Footer component
 */

import { APP_NAME } from "@/lib/constants/config";
import { Mail } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-border/40 bg-background">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex h-16 items-center justify-between">
        <div className="text-sm text-muted-foreground">
          © {currentYear} {APP_NAME}. All rights reserved.
        </div>
        
        {/* Disclaimer */}
        <div className="text-xs text-muted-foreground text-center flex-1 mx-4">
          PoEconomy is not affiliated with Grinding Gear Games.
        </div>
        
        {/* Contact Information */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Mail className="h-4 w-4" />
          <a 
            href="mailto:contact@poeconomy.com" 
            className="hover:text-foreground transition-colors"
          >
            contact@poeconomy.com
          </a>
        </div>
      </div>
    </footer>
  );
}

