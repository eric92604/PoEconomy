const WINDOW_SECONDS = 60;

export async function kvRateLimit(request: Request, kv: KVNamespace, maxPerMinute: number): Promise<Response | null> {
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const now = Math.floor(Date.now() / 1000);
  const window = Math.floor(now / WINDOW_SECONDS);
  const windowKey = ip + ':' + window;
  const current = await kv.get(windowKey);
  const count = current ? parseInt(current, 10) : 0;

  if (count >= maxPerMinute) {
    return new Response('Rate limit exceeded', {
      status: 429,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Retry-After': '60',
      },
    });
  }

  await kv.put(windowKey, String(count + 1), { expirationTtl: WINDOW_SECONDS });
  return null;
}
