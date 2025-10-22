/**
 * Service Worker for caching currency icons and API responses
 */

const CACHE_NAME = 'poeconomy-v1';
const ICON_CACHE_NAME = 'poeconomy-icons-v1';
const API_CACHE_NAME = 'poeconomy-api-v1';

// Cache duration in milliseconds
const CACHE_DURATIONS = {
  icons: 7 * 24 * 60 * 60 * 1000, // 7 days
  api: 5 * 60 * 1000, // 5 minutes
  static: 24 * 60 * 60 * 1000, // 24 hours
};

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('Service Worker installing...');
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME && cacheName !== ICON_CACHE_NAME && cacheName !== API_CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle currency icons
  if (url.pathname.startsWith('/images/currency/')) {
    event.respondWith(handleIconRequest(request));
    return;
  }

  // Handle API requests
  if (url.pathname.startsWith('/api/') || url.hostname.includes('api')) {
    event.respondWith(handleApiRequest(request));
    return;
  }

  // Handle static assets
  if (url.pathname.startsWith('/_next/static/') || url.pathname.endsWith('.js') || url.pathname.endsWith('.css')) {
    event.respondWith(handleStaticRequest(request));
    return;
  }
});

/**
 * Handle currency icon requests with cache-first strategy
 */
async function handleIconRequest(request) {
  const cache = await caches.open(ICON_CACHE_NAME);
  const cachedResponse = await cache.match(request);

  if (cachedResponse) {
    // Check if cache is still valid
    const cacheDate = cachedResponse.headers.get('sw-cache-date');
    if (cacheDate && Date.now() - parseInt(cacheDate) < CACHE_DURATIONS.icons) {
      return cachedResponse;
    }
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      // Clone response and add cache headers
      const responseToCache = response.clone();
      responseToCache.headers.set('sw-cache-date', Date.now().toString());
      cache.put(request, responseToCache);
    }
    return response;
  } catch (error) {
    // Return cached version if available, even if stale
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}

/**
 * Handle API requests with stale-while-revalidate strategy
 */
async function handleApiRequest(request) {
  const cache = await caches.open(API_CACHE_NAME);
  const cachedResponse = await cache.match(request);

  // Return cached response immediately if available
  if (cachedResponse) {
    const cacheDate = cachedResponse.headers.get('sw-cache-date');
    if (cacheDate && Date.now() - parseInt(cacheDate) < CACHE_DURATIONS.api) {
      // Update cache in background
      fetch(request).then((response) => {
        if (response.ok) {
          const responseToCache = response.clone();
          responseToCache.headers.set('sw-cache-date', Date.now().toString());
          cache.put(request, responseToCache);
        }
      }).catch(() => {
        // Ignore background update errors
      });
      
      return cachedResponse;
    }
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      const responseToCache = response.clone();
      responseToCache.headers.set('sw-cache-date', Date.now().toString());
      cache.put(request, responseToCache);
    }
    return response;
  } catch (error) {
    // Return stale cache if available
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}

/**
 * Handle static asset requests with cache-first strategy
 */
async function handleStaticRequest(request) {
  const cache = await caches.open(CACHE_NAME);
  const cachedResponse = await cache.match(request);

  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response);
    }
    return response;
  } catch (error) {
    throw error;
  }
}
