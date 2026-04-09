import { kvRateLimit } from './rateLimit';

interface Env {
  AWS_API_GATEWAY_URL: string;
  AWS_API_KEY: string;
  CACHE_TTL?: string;
  RATE_LIMIT_PER_MINUTE?: string;
  DEBUG_MODE?: string;
  CACHE_WARM_SECRET?: string;
  CLOUDFLARE_WORKER_URL?: string;
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
  '/predict/batch',
  '/prices/leagues',        // league list from historical archive table
  '/prices/league-history', // cross-league price comparison series
  '/prices/history',        // daily aggregated prices for a single league
]);

async function handleCacheWarm(request: Request, env: Env): Promise<Response> {
  const secret = request.headers.get('x-cache-warm-secret');
  if (!secret || secret !== env.CACHE_WARM_SECRET) {
    return new Response(JSON.stringify({ error: 'Unauthorized' }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const body = await request.json() as { league?: string; leagues?: string[] };
  const leagues: string[] = body.leagues ?? (body.league ? [body.league] : []);

  const warmed: string[] = [];
  const failed: string[] = [];
  const baseUrl = env.CLOUDFLARE_WORKER_URL ?? 'https://api.poeconomy.com';

  // Always warm static metadata endpoints
  const staticEndpoints = ['/predict/currencies', '/predict/leagues'];
  for (const path of staticEndpoints) {
    try {
      const resp = await fetch(`${baseUrl}${path}`);
      if (resp.ok) warmed.push(path);
      else failed.push(path);
    } catch {
      failed.push(path);
    }
  }

  // Warm per-league prediction endpoints
  for (const league of leagues) {
    const paths = [
      `/predict/latest?league=${encodeURIComponent(league)}&horizons=1d,3d,7d`,
    ];
    for (const path of paths) {
      try {
        const resp = await fetch(`${baseUrl}${path}`);
        if (resp.ok) warmed.push(path);
        else failed.push(path);
      } catch {
        failed.push(path);
      }
    }
  }

  return new Response(JSON.stringify({ warmed, failed }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

async function handleCacheWarmHistory(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const secret = request.headers.get('x-cache-warm-secret');
  if (!secret || secret !== env.CACHE_WARM_SECRET) {
    return new Response(JSON.stringify({ error: 'Unauthorized' }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const baseUrl = env.CLOUDFLARE_WORKER_URL ?? 'https://api.poeconomy.com';

  // Fetch leagues and currencies in parallel, then fan out to all (currency, league) pairs.
  // Each subrequest hits this same Worker so the existing KV caching logic fires automatically.
  // Returns 202 immediately; the actual warming runs in the background via ctx.waitUntil.
  async function warmAllPairs(): Promise<void> {
    // 1. Enumerate leagues from /prices/leagues
    let leagues: string[] = [];
    try {
      const leaguesResp = await fetch(`${baseUrl}/prices/leagues`);
      if (leaguesResp.ok) {
        const data = await leaguesResp.json() as { leagues: Record<string, unknown> };
        leagues = Object.keys(data.leagues ?? {});
      }
    } catch (e) {
      console.error('[cache-warm-history] Failed to fetch leagues:', e);
    }

    // 2. Enumerate currencies from /predict/currencies
    let currencies: string[] = [];
    try {
      const currResp = await fetch(`${baseUrl}/predict/currencies`);
      if (currResp.ok) {
        const data = await currResp.json() as { currencies: Record<string, unknown> };
        // currencies is keyed by league; collect unique currency names across all leagues
        const currencySet = new Set<string>();
        for (const leagueCurrencies of Object.values(data.currencies ?? {})) {
          for (const name of Object.keys(leagueCurrencies as Record<string, unknown>)) {
            currencySet.add(name);
          }
        }
        currencies = Array.from(currencySet);
      }
    } catch (e) {
      console.error('[cache-warm-history] Failed to fetch currencies:', e);
    }

    if (leagues.length === 0 || currencies.length === 0) {
      console.error('[cache-warm-history] No leagues or currencies found, aborting.');
      return;
    }

    // 3. Build all (currency, league) pairs
    const pairs: { currency: string; league: string }[] = [];
    for (const league of leagues) {
      for (const currency of currencies) {
        pairs.push({ currency, league });
      }
    }

    console.log(`[cache-warm-history] Warming ${pairs.length} (currency, league) pairs across ${leagues.length} leagues and ${currencies.length} currencies`);

    // 4. Fan out in batches of 10 to avoid overwhelming the origin Lambda
    const BATCH_SIZE = 10;
    let warmed = 0;
    let failed = 0;
    for (let i = 0; i < pairs.length; i += BATCH_SIZE) {
      const batch = pairs.slice(i, i + BATCH_SIZE);
      await Promise.all(
        batch.map(async ({ currency, league }) => {
          const path = `/prices/league-history?currency=${encodeURIComponent(currency)}&league=${encodeURIComponent(league)}`;
          try {
            const resp = await fetch(`${baseUrl}${path}`);
            if (resp.ok) {
              warmed++;
            } else {
              failed++;
              console.warn(`[cache-warm-history] Non-OK response for ${path}: ${resp.status}`);
            }
          } catch (e) {
            failed++;
            console.error(`[cache-warm-history] Fetch failed for ${path}:`, e);
          }
        })
      );
    }

    console.log(`[cache-warm-history] Complete — warmed: ${warmed}, failed: ${failed}, total: ${pairs.length}`);
  }

  ctx.waitUntil(warmAllPairs());

  return new Response(
    JSON.stringify({
      status: 'accepted',
      message: 'Historical cache warm started in background. Check Worker logs for progress.',
    }),
    { status: 202, headers: { 'Content-Type': 'application/json' } }
  );
}

async function handleCacheClear(request: Request, env: Env): Promise<Response> {
  const secret = request.headers.get('x-cache-warm-secret');
  if (!secret || secret !== env.CACHE_WARM_SECRET) {
    return new Response(JSON.stringify({ error: 'Unauthorized' }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const body = await request.json() as { prefix?: string };
  const prefix = body.prefix ?? '';

  let deleted = 0;
  let cursor: string | undefined;
  do {
    const list = await env.CACHE_KV.list({ prefix, cursor, limit: 1000 });
    for (const key of list.keys) {
      await env.CACHE_KV.delete(key.name);
      deleted++;
    }
    cursor = list.list_complete ? undefined : list.cursor;
  } while (cursor);

  return new Response(JSON.stringify({ deleted, prefix }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

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

    // Admin endpoints — bypass rate limiting
    if (url.pathname === '/admin/cache-warm' && request.method === 'POST') {
      return handleCacheWarm(request, env);
    }
    if (url.pathname === '/admin/cache-warm-history' && request.method === 'POST') {
      return handleCacheWarmHistory(request, env, ctx);
    }
    if (url.pathname === '/admin/cache-clear' && request.method === 'POST') {
      return handleCacheClear(request, env);
    }

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
      const cacheKey = `${url.pathname}${url.search}`;
      const cached = await env.CACHE_KV.get(cacheKey, 'json');
      if (cached) {
        const { body, init } = cached as CachedResponse;
        const headers = new Headers(init.headers);
        headers.set('Access-Control-Allow-Origin', '*');
        headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-api-key');
        headers.set('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');

        let browserCacheTTL: number;
        if (url.pathname === '/predict/currencies' || url.pathname === '/predict/leagues') {
          browserCacheTTL = 1800;
        } else if (url.pathname === '/prices/leagues' || url.pathname === '/prices/league-history') {
          browserCacheTTL = 86400;
        } else if (url.pathname === '/prices/history') {
          browserCacheTTL = 3600;
        } else if (url.pathname.includes('/predict/')) {
          browserCacheTTL = 300;
        } else {
          browserCacheTTL = 600;
        }

        headers.set(
          'Cache-Control',
          `public, max-age=${browserCacheTTL}, stale-while-revalidate=${browserCacheTTL * 2}`
        );
        headers.set('X-Cache', 'HIT');

        return new Response(JSON.stringify(body), { ...init, headers });
      }
    }

    // Create headers with API key - only send essential headers to AWS
    const upstreamHeaders = new Headers();
    upstreamHeaders.set('x-api-key', env.AWS_API_KEY);
    upstreamHeaders.set('Content-Type', 'application/json');

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

      console.log('=== AWS API RESPONSE ===');
      console.log('Status:', response.status);
      console.log('Body:', respBody);
    } catch (error) {
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
    headers.set('X-Cache', 'MISS');

    if (isCacheable && response.ok) {
      const cacheKey = `${url.pathname}${url.search}`;

      let ttl: number | null;
      let browserCacheTTL: number;

      if (url.pathname === '/predict/latest') {
        ttl = 600;
        browserCacheTTL = 300;
      } else if (url.pathname === '/predict/currency') {
        ttl = 600;
        browserCacheTTL = 300;
      } else if (url.pathname === '/predict/batch') {
        ttl = 600;
        browserCacheTTL = 300;
      } else if (url.pathname === '/predict/currencies' || url.pathname === '/predict/leagues') {
        ttl = Number(env.CACHE_TTL ?? '1800');
        browserCacheTTL = 1800;
      } else if (url.pathname === '/prices/leagues' || url.pathname === '/prices/league-history') {
        ttl = null;     // permanent — immutable seeded data, cleared manually on new league
        browserCacheTTL = 86400;
      } else if (url.pathname === '/prices/history') {
        ttl = 86400;    // 24h — daily aggregation appends one new row per day
        browserCacheTTL = 3600;
      } else {
        ttl = Number(env.CACHE_TTL ?? '1800');
        browserCacheTTL = 600;
      }

      headers.set(
        'Cache-Control',
        `public, max-age=${browserCacheTTL}, stale-while-revalidate=${browserCacheTTL * 2}`
      );

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

      const putOptions = ttl !== null ? { expirationTtl: ttl } : {};
      await env.CACHE_KV.put(cacheKey, JSON.stringify(cacheData), putOptions);
    } else if (!response.ok) {
      headers.set('Cache-Control', 'no-store');
    }

    return new Response(respBody, { status: response.status, headers });
  },
};
