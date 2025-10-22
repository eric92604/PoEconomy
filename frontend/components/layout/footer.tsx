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
        <div className="text-xs text-muted-foreground text-right">
          This application is not affiliated with Grinding Gear Games. Path of Exile is a trademark of Grinding Gear Games Ltd.
        </div>
      </div>
    </footer>
  );
}

