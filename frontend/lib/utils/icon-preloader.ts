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
 * Preload all currency icons - preloads local AVIF icons for better performance
 */
export async function preloadAllCurrencyIcons(currencyData: Record<string, Record<string, { icon_url?: string }>>): Promise<void> {
  const localIconUrls: string[] = [];
  
  // Get all unique currency names to preload their local AVIF icons
  const currencyNames = new Set<string>();
  Object.keys(currencyData).forEach(currencyName => {
    currencyNames.add(currencyName);
  });

  // Import the currency icon mapping to get local paths
  const { getOptimizedCurrencyIcon } = await import('@/lib/constants/currency-icons');
  
  currencyNames.forEach(currencyName => {
    const localIconPath = getOptimizedCurrencyIcon(currencyName);
    if (localIconPath) {
      localIconUrls.push(localIconPath);
    }
  });

  if (localIconUrls.length > 0) {
    console.log(`🔄 Preloading ${localIconUrls.length} local AVIF currency icons...`);
    await iconPreloader.preloadIcons(localIconUrls, { priority: true });
    console.log('✅ Local AVIF currency icons preloaded');
  } else {
    console.log('⚠️ No local currency icons found to preload');
  }
}

/**
 * Preload icons for visible currencies - uses local AVIF icons
 */
export async function preloadVisibleIcons(
  currencies: Array<{ currency: string; icon_url?: string }>,
  options: PreloadOptions = {}
): Promise<void> {
  const localIconUrls: string[] = [];
  
  // Import the currency icon mapping to get local paths
  const { getOptimizedCurrencyIcon } = await import('@/lib/constants/currency-icons');
  
  currencies.forEach(currency => {
    const localIconPath = getOptimizedCurrencyIcon(currency.currency);
    if (localIconPath) {
      localIconUrls.push(localIconPath);
    }
  });

  if (localIconUrls.length > 0) {
    await iconPreloader.preloadIcons(localIconUrls, options);
  }
}
