/**
 * Icon preloading utilities for currency icons
 */

interface PreloadOptions {
  priority?: boolean;
  timeout?: number;
}

class IconPreloader {
  private preloadedIcons = new Set<string>();
  private preloadPromises = new Map<string, Promise<void>>();

  /**
   * Preload a single icon
   */
  async preloadIcon(iconUrl: string, options: PreloadOptions = {}): Promise<void> {
    if (this.preloadedIcons.has(iconUrl)) {
      return Promise.resolve();
    }

    if (this.preloadPromises.has(iconUrl)) {
      return this.preloadPromises.get(iconUrl)!;
    }

    const promise = new Promise<void>((resolve, reject) => {
      const img = new Image();
      
      const timeout = setTimeout(() => {
        reject(new Error(`Icon preload timeout: ${iconUrl}`));
      }, options.timeout || 5000);

      img.onload = () => {
        clearTimeout(timeout);
        this.preloadedIcons.add(iconUrl);
        resolve();
      };

      img.onerror = () => {
        clearTimeout(timeout);
        reject(new Error(`Failed to preload icon: ${iconUrl}`));
      };

      img.src = iconUrl;
    });

    this.preloadPromises.set(iconUrl, promise);
    return promise;
  }

  /**
   * Preload multiple icons in parallel
   */
  async preloadIcons(iconUrls: string[], options: PreloadOptions = {}): Promise<void[]> {
    const promises = iconUrls.map(url => 
      this.preloadIcon(url, options).catch(error => {
        console.warn(`Failed to preload icon: ${url}`, error);
        return Promise.resolve();
      })
    );

    return Promise.all(promises);
  }

  /**
   * Check if an icon is already preloaded
   */
  isPreloaded(iconUrl: string): boolean {
    return this.preloadedIcons.has(iconUrl);
  }

  /**
   * Clear preload cache
   */
  clearCache(): void {
    this.preloadedIcons.clear();
    this.preloadPromises.clear();
  }
}

// Singleton instance
export const iconPreloader = new IconPreloader();

/**
 * Critical currency icons that should be preloaded immediately
 */
export const CRITICAL_CURRENCY_ICONS = [
  'Divine Orb',
  'Exalted Orb', 
  'Chaos Orb',
  'Mirror of Kalandra',
  'Orb of Fusing',
  'Chromatic Orb',
  'Orb of Alchemy',
  'Vaal Orb'
];

/**
 * Preload critical currency icons
 */
export async function preloadCriticalIcons(currencyData: any): Promise<void> {
  const criticalUrls: string[] = [];
  
  CRITICAL_CURRENCY_ICONS.forEach(currency => {
    const metadata = currencyData.currencies?.[currency];
    if (metadata) {
      Object.values(metadata).forEach((leagueData: any) => {
        if (leagueData?.icon_url) {
          criticalUrls.push(leagueData.icon_url);
        }
      });
    }
  });

  if (criticalUrls.length > 0) {
    await iconPreloader.preloadIcons(criticalUrls, { priority: true });
  }
}

/**
 * Preload icons for visible currencies
 */
export async function preloadVisibleIcons(
  currencies: Array<{ icon_url?: string }>,
  options: PreloadOptions = {}
): Promise<void> {
  const iconUrls = currencies
    .map(c => c.icon_url)
    .filter((url): url is string => Boolean(url));

  if (iconUrls.length > 0) {
    await iconPreloader.preloadIcons(iconUrls, options);
  }
}
