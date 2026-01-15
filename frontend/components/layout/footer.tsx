/**
 * Footer component
 */

import { APP_NAME } from "@/lib/constants/config";
import { Mail } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-border/40 bg-card/30">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="text-sm text-muted-foreground">
            © {currentYear} {APP_NAME}. All rights reserved.
          </div>
          
          <div className="text-xs text-muted-foreground text-center">
            PoEconomy is not affiliated with Grinding Gear Games.
          </div>
          
          <a 
            href="mailto:contact@poeconomy.com" 
            className="text-sm text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-1.5"
          >
            <Mail className="h-3.5 w-3.5" />
            contact@poeconomy.com
          </a>
        </div>
      </div>
    </footer>
  );
}

