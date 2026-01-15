/**
 * Footer component with POE community links
 */

import { APP_NAME } from "@/lib/constants/config";
import { Mail, ExternalLink } from "lucide-react";
import Image from "next/image";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-border/40 bg-card/30">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Image
                src="/images/mirror-of-kalandra.png"
                alt="PoEconomy"
                width={24}
                height={24}
              />
              <span className="font-bold text-lg">{APP_NAME}</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Your edge in Wraeclast&apos;s economy. ML-powered currency predictions for Path of Exile.
            </p>
          </div>

          {/* Quick Links */}
          <div className="space-y-3">
            <h3 className="font-semibold text-sm text-poe-gold">Quick Links</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="/dashboard" className="text-muted-foreground hover:text-foreground transition-colors">
                  Dashboard
                </a>
              </li>
              <li>
                <a href="/investments" className="text-muted-foreground hover:text-foreground transition-colors">
                  Investments
                </a>
              </li>
              <li>
                <a href="/prices" className="text-muted-foreground hover:text-foreground transition-colors">
                  Live Prices
                </a>
              </li>
            </ul>
          </div>

          {/* Community */}
          <div className="space-y-3">
            <h3 className="font-semibold text-sm text-poe-gold">Community</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a 
                  href="https://www.pathofexile.com/trade" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-1"
                >
                  Official Trade Site
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li>
                <a 
                  href="https://www.pathofexile.com" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-1"
                >
                  Path of Exile
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div className="space-y-3">
            <h3 className="font-semibold text-sm text-poe-gold">Contact</h3>
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
        </div>

        {/* Bottom Bar */}
        <div className="pt-6 border-t border-border/40 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-sm text-muted-foreground">
            © {currentYear} {APP_NAME}. All rights reserved.
          </div>
          
          <div className="text-xs text-muted-foreground text-center">
            PoEconomy is a fan project and is not affiliated with Grinding Gear Games.
          </div>
        </div>
      </div>
    </footer>
  );
}

