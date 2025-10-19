/**
 * Footer component
 */

import { APP_NAME, APP_VERSION } from "@/lib/constants/config";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-border/40 bg-background">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex h-16 items-center justify-between">
        <div className="text-sm text-muted-foreground">
          © {currentYear} {APP_NAME}. All rights reserved.
        </div>
        <div className="flex items-center space-x-4 text-sm text-muted-foreground">
          <span>v{APP_VERSION}</span>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors"
          >
            GitHub
          </a>
          <a
            href="/docs"
            className="hover:text-foreground transition-colors"
          >
            Docs
          </a>
        </div>
      </div>
    </footer>
  );
}

