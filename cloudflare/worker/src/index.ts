import { kvRateLimit } from './rateLimit';

interface Env {
  AWS_API_GATEWAY_URL: string;
  AWS_API_KEY: string;
  CACHE_TTL?: string;
  RATE_LIMIT_PER_MINUTE?: string;
  DEBUG_MODE?: string;
  CACHE_KV: KVNamespace;
  RATE_LIMIT_KV: KVNamespace;
}

interface CachedResponse {
  body: unknown;
  init: ResponseInit;
}

const CACHEABLE_PATHS = new Set([
  '/predict/currencies', 
  '/predict/leagues',
  '/predict/latest',
  '/predict/currency',
  '/predict/batch'
]);

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
          'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        },
      });
    }

    const url = new URL(request.url);
    const perMinute = Number(env.RATE_LIMIT_PER_MINUTE ?? '60');
    if (perMinute > 0) {
      const rateResp = await kvRateLimit(request, env.RATE_LIMIT_KV, perMinute);
      if (rateResp) return rateResp;
    }

    const targetUrl = new URL(env.AWS_API_GATEWAY_URL);
    targetUrl.pathname += url.pathname;
    targetUrl.search = url.search;

    const isCacheable = request.method === 'GET' && CACHEABLE_PATHS.has(url.pathname);

    if (isCacheable) {
      // Generate cache key from URL and query parameters
      const cacheKey = `${url.pathname}${url.search}`;
      const cached = await env.CACHE_KV.get(cacheKey, 'json');
      if (cached) {
        const { body, init } = cached as CachedResponse;
        const headers = new Headers(init.headers);
        headers.set('Access-Control-Allow-Origin', '*');
        headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-api-key');
        headers.set('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
        
        // Add cache headers for cached responses too
        let browserCacheTTL: number;
        if (url.pathname === '/predict/currencies' || url.pathname === '/predict/leagues') {
          browserCacheTTL = 1800; // 30 minutes for metadata
        } else if (url.pathname.includes('/predict/')) {
          browserCacheTTL = 600; // 10 minutes for predictions
        } else {
          browserCacheTTL = 600; // 10 minutes default
        }
        
        headers.set(
          'Cache-Control', 
          `public, max-age=${browserCacheTTL}, stale-while-revalidate=${browserCacheTTL * 2}`
        );
        headers.set('X-Cache', 'HIT'); // Helpful for debugging
        
        return new Response(JSON.stringify(body), { ...init, headers });
      }
    }

    // Create headers with API key - only send essential headers to AWS
    const upstreamHeaders = new Headers();
    upstreamHeaders.set('x-api-key', env.AWS_API_KEY);
    upstreamHeaders.set('Content-Type', 'application/json');
    
    // Only forward the request body if it exists
    if (request.body) {
      upstreamHeaders.set('Content-Length', request.headers.get('content-length') || '0');
    }

    const debugMode = env.DEBUG_MODE === 'true';
    
    if (debugMode) {
      console.log('=== REQUEST DEBUG ===');
      console.log('Method:', request.method);
      console.log('Original URL:', request.url);
      console.log('Target URL:', targetUrl.toString());
      console.log('API Key (first 10 chars):', env.AWS_API_KEY?.substring(0, 10) + '...');
      const headersObj: Record<string, string> = {};
      upstreamHeaders.forEach((value, key) => { headersObj[key] = value; });
      console.log('Headers sent:', headersObj);
    }

    const upstreamRequest = new Request(targetUrl.toString(), {
      method: request.method,
      headers: upstreamHeaders,
      body: request.body,
    });
    
    if (debugMode) {
      console.log('=== UPSTREAM REQUEST ===');
      console.log('URL:', upstreamRequest.url);
      console.log('Method:', upstreamRequest.method);
      const upstreamHeadersObj: Record<string, string> = {};
      upstreamRequest.headers.forEach((value, key) => { upstreamHeadersObj[key] = value; });
      console.log('Headers:', upstreamHeadersObj);
    }

    let response: Response;
    let respBody: string;
    
    try {
      response = await fetch(upstreamRequest);
      respBody = await response.text();
      
      if (debugMode) {
        console.log('=== RESPONSE DEBUG ===');
        console.log('Status:', response.status, response.statusText);
        const respHeadersObj: Record<string, string> = {};
        response.headers.forEach((value, key) => { respHeadersObj[key] = value; });
        console.log('Headers received:', respHeadersObj);
        console.log('Body:', respBody.length > 500 ? respBody.substring(0, 500) + '...' : respBody);
      }
      
      // Log the actual response for debugging
      console.log('=== AWS API RESPONSE ===');
      console.log('Status:', response.status);
      console.log('Body:', respBody);
    } catch (error) {
      // Always log errors (critical for debugging)
      console.error('=== FETCH ERROR ===');
      console.error('Error:', error);
      return new Response(JSON.stringify({ 
        error: 'Failed to connect to API Gateway',
        details: debugMode && error instanceof Error ? error.message : 'Connection failed',
        targetUrl: debugMode ? targetUrl.toString() : undefined
      }), {
        status: 502,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        }
      });
    }

    const headers = new Headers(response.headers);
    headers.set('Access-Control-Allow-Origin', '*');
    headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-api-key');
    headers.set('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    headers.set('X-Cache', 'MISS'); // Helpful for debugging - fresh from origin

    if (isCacheable && response.ok) {
      // Generate cache key for storage
      const cacheKey = `${url.pathname}${url.search}`;
      
      // Set different TTL based on endpoint type
      let ttl: number;
      let browserCacheTTL: number;
      
      if (url.pathname === '/predict/latest') {
        ttl = 600; // 10 minutes for latest predictions (optimal for fresh data)
        browserCacheTTL = 300; // 5 minutes browser cache
      } else if (url.pathname === '/predict/currency') {
        ttl = 600; // 10 minutes for currency-specific predictions (optimal for fresh data)
        browserCacheTTL = 300; // 5 minutes browser cache
      } else if (url.pathname === '/predict/batch') {
        ttl = 600; // 10 minutes for batch predictions (optimal for fresh data)
        browserCacheTTL = 300; // 5 minutes browser cache
      } else if (url.pathname === '/predict/currencies' || url.pathname === '/predict/leagues') {
        ttl = Number(env.CACHE_TTL ?? '1800'); // 30 minutes for metadata endpoints
        browserCacheTTL = 1800; // 30 minutes browser cache - metadata changes rarely
      } else {
        ttl = Number(env.CACHE_TTL ?? '1800'); // 1 hour for other endpoints
        browserCacheTTL = 600; // 10 minutes browser cache
      }
      
      // Add Cache-Control headers for browser caching
      // stale-while-revalidate allows serving stale content while fetching fresh data
      headers.set(
        'Cache-Control', 
        `public, max-age=${browserCacheTTL}, stale-while-revalidate=${browserCacheTTL * 2}`
      );
      
      // Add ETag for conditional requests (optional but recommended)
      const etag = `W/"${Date.now()}-${cacheKey.substring(0, 8)}"`;
      headers.set('ETag', etag);
      
      const headersArray: [string, string][] = [];
      headers.forEach((value, key) => { headersArray.push([key, value]); });
      
      const cacheData: CachedResponse = {
        body: JSON.parse(respBody),
        init: {
          status: response.status,
          headers: headersArray,
        },
      };
      await env.CACHE_KV.put(cacheKey, JSON.stringify(cacheData), { expirationTtl: ttl });
    } else if (!response.ok) {
      // Don't cache errors in browser
      headers.set('Cache-Control', 'no-store');
    }

    return new Response(respBody, { status: response.status, headers });
  },
};
