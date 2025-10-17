import { kvRateLimit } from './rateLimit';

interface Env {
  AWS_API_GATEWAY_URL: string;
  CACHE_TTL?: string;
  RATE_LIMIT_PER_MINUTE?: string;
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
  '/predict/currency'
]);

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
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
    targetUrl.pathname = url.pathname;
    targetUrl.search = url.search;

    const cacheKey = request.method + '-' + targetUrl.toString();
    const isCacheable = request.method === 'GET' && CACHEABLE_PATHS.has(url.pathname);

    if (isCacheable) {
      const cached = await env.CACHE_KV.get(cacheKey, 'json');
      if (cached) {
        const { body, init } = cached as CachedResponse;
        const headers = new Headers(init.headers);
        headers.set('Access-Control-Allow-Origin', '*');
        headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
        headers.set('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
        return new Response(JSON.stringify(body), { ...init, headers });
      }
    }

    const upstreamRequest = new Request(targetUrl.toString(), {
      method: request.method,
      headers: request.headers,
      body: request.body,
    });

    const response = await fetch(upstreamRequest);
    const respBody = await response.text();

    const headers = new Headers(response.headers);
    headers.set('Access-Control-Allow-Origin', '*');
    headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    headers.set('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');

    if (isCacheable && response.ok) {
      // Set different TTL based on endpoint type
      let ttl: number;
      if (url.pathname === '/predict/latest') {
        ttl = 600; // 10 minutes for latest predictions (optimal for fresh data)
      } else if (url.pathname === '/predict/currency') {
        ttl = 600; // 10 minutes for currency-specific predictions (optimal for fresh data)
      } else {
        ttl = Number(env.CACHE_TTL ?? '300'); // 5 minutes for metadata endpoints
      }
      
      const cacheData: CachedResponse = {
        body: JSON.parse(respBody),
        init: {
          status: response.status,
          headers: [...headers.entries()],
        },
      };
      await env.CACHE_KV.put(cacheKey, JSON.stringify(cacheData), { expirationTtl: ttl });
    }

    return new Response(respBody, { status: response.status, headers });
  },
};
