/**
 * Service Worker utilities for registration and management
 */

export function registerServiceWorker() {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return Promise.resolve();
  }

  return navigator.serviceWorker
    .register('/sw.js')
    .then((registration) => {
      console.log('Service Worker registered successfully:', registration);
      return registration;
    })
    .catch((error) => {
      console.error('Service Worker registration failed:', error);
      throw error;
    });
}

export function unregisterServiceWorker() {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return Promise.resolve();
  }

  return navigator.serviceWorker
    .getRegistrations()
    .then((registrations) => {
      return Promise.all(
        registrations.map((registration) => registration.unregister())
      );
    });
}

export function clearAllCaches() {
  if (typeof window === 'undefined' || !('caches' in window)) {
    return Promise.resolve();
  }

  return caches.keys().then((cacheNames) => {
    return Promise.all(
      cacheNames.map((cacheName) => caches.delete(cacheName))
    );
  });
}
