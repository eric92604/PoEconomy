/**
 * PoEconomy Cloudflare Worker
 * 
 * This worker provides:
 * - Edge caching for ML predictions
 * - Rate limiting and DDoS protection
 * - Request routing and proxying to AWS API Gateway
 * - CORS handling
 * - Error handling and logging
 */

// Environment variables that need to be set in Cloudflare Workers
interface Env {
  AWS_API_GATEWAY_URL: string;
  RATE_LIMIT_KV: KVNamespace;
  CACHE_KV: KVNamespace;
  API_KEY?: string;
  ENVIRONMENT: string;
}

// Types
interface PredictionRequest {
  currency: string;
  prediction_horizon?: string;
}

interface BatchPredictionRequest {
  currencies: string[];
  prediction_horizon?: string;
}

interface RateLimitInfo {
  count: number;
  resetTime: number;
}

interface CacheItem {
  data: any;
  timestamp: number;
  ttl: number;
}

// Configuration
const CONFIG = {
  RATE_LIMIT: {
    REQUESTS_PER_MINUTE: 60,
    REQUESTS_PER_HOUR: 1000,
    BURST_LIMIT: 10,
  },
  CACHE_TTL: {
    PREDICTIONS: 900, // 15 minutes for predictions (matches data update frequency)
    RECOMMENDATIONS: 900, // 15 minutes for recommendations (same as predictions)
    HEALTH: 60, // 1 minute for health checks
  },
  CORS_HEADERS: {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
    'Access-Control-Max-Age': '86400',
  },
  AWS_TIMEOUT: 30000, // 30 seconds
};

/**
 * Main request handler
 */
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    try {
      // Handle CORS preflight requests
      if (request.method === 'OPTIONS') {
        return handleCORS();
      }

      // Get client IP for rate limiting
      const clientIP = request.headers.get('CF-Connecting-IP') || 'unknown';
      
      // Check rate limits
      const rateLimitResult = await checkRateLimit(clientIP, env.RATE_LIMIT_KV);
      if (!rateLimitResult.allowed) {
        return createErrorResponse(429, 'Rate limit exceeded', {
          'X-RateLimit-Limit': CONFIG.RATE_LIMIT.REQUESTS_PER_MINUTE.toString(),
          'X-RateLimit-Remaining': '0',
          'X-RateLimit-Reset': rateLimitResult.resetTime?.toString() || '',
        });
      }

      // Parse URL and route request
      const url = new URL(request.url);
      const path = url.pathname;

      // Health check endpoint
      if (path === '/health' || path === '/') {
        return handleHealthCheck(env);
      }

      // API endpoints
      if (path.startsWith('/api/') || path.startsWith('/predict')) {
        return await handleAPIRequest(request, env, ctx, clientIP);
      }

      // Default 404
      return createErrorResponse(404, 'Endpoint not found');

    } catch (error) {
      console.error('Worker error:', error);
      return createErrorResponse(500, 'Internal server error');
    }
  },
};

/**
 * Handle CORS preflight requests
 */
function handleCORS(): Response {
  return new Response(null, {
    status: 204,
    headers: CONFIG.CORS_HEADERS,
  });
}

/**
 * Handle health check requests
 */
async function handleHealthCheck(env: Env): Promise<Response> {
  const cacheKey = 'health_check';
  
  // Try to get cached health status
  const cached = await getCachedItem(env.CACHE_KV, cacheKey);
  if (cached) {
    return createSuccessResponse(cached.data);
  }

  // Perform health check
  const healthData = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    worker_version: '1.0.0',
    environment: env.ENVIRONMENT || 'production',
    uptime: Date.now(),
  };

  // Cache health status
  await setCachedItem(env.CACHE_KV, cacheKey, healthData, CONFIG.CACHE_TTL.HEALTH);

  return createSuccessResponse(healthData);
}

/**
 * Handle API requests (predictions, recommendations, etc.)
 */
async function handleAPIRequest(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  clientIP: string
): Promise<Response> {
  const url = new URL(request.url);
  let path = url.pathname;
  
  // Normalize path
  if (path.startsWith('/api')) {
    path = path.substring(4);
  }

  // Generate cache key based on request
  const cacheKey = await generateCacheKey(request, path);
  
  // Try to get cached response
  const cached = await getCachedItem(env.CACHE_KV, cacheKey);
  if (cached) {
    console.log(`Cache hit for key: ${cacheKey}`);
    return createSuccessResponse(cached.data, {
      'X-Cache': 'HIT',
      'X-Cache-TTL': (cached.ttl - (Date.now() - cached.timestamp) / 1000).toString(),
    });
  }

  // Forward request to AWS API Gateway
  try {
    const awsResponse = await forwardToAWS(request, env, path);
    const responseData = await awsResponse.json();

    // Determine cache TTL based on endpoint
    let cacheTTL = 0;
    if (path.includes('/predict')) {
      cacheTTL = CONFIG.CACHE_TTL.PREDICTIONS;
    } else if (path.includes('/recommendations')) {
      cacheTTL = CONFIG.CACHE_TTL.RECOMMENDATIONS;
    }

    // Cache successful responses
    if (awsResponse.ok && cacheTTL > 0) {
      await setCachedItem(env.CACHE_KV, cacheKey, responseData, cacheTTL);
    }

    // Update rate limit counter
    await updateRateLimit(clientIP, env.RATE_LIMIT_KV);

    return createSuccessResponse(responseData, {
      'X-Cache': 'MISS',
      'X-Response-Time': awsResponse.headers.get('X-Response-Time') || '',
    });

  } catch (error) {
    console.error('AWS request failed:', error);
    return createErrorResponse(502, 'Backend service unavailable');
  }
}

/**
 * Forward request to AWS API Gateway
 */
async function forwardToAWS(request: Request, env: Env, path: string): Promise<Response> {
  const awsUrl = `${env.AWS_API_GATEWAY_URL}${path}`;
  
  // Prepare headers
  const headers = new Headers(request.headers);
  headers.set('User-Agent', 'PoEconomy-CloudflareWorker/1.0');
  
  // Add API key if available
  if (env.API_KEY) {
    headers.set('X-API-Key', env.API_KEY);
  }

  // Forward request
  const awsRequest = new Request(awsUrl, {
    method: request.method,
    headers,
    body: request.method !== 'GET' ? await request.blob() : null,
  });

  // Set timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), CONFIG.AWS_TIMEOUT);

  try {
    const response = await fetch(awsRequest, {
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Check rate limits
 */
async function checkRateLimit(clientIP: string, kv: KVNamespace): Promise<{
  allowed: boolean;
  resetTime?: number;
}> {
  const now = Date.now();
  const minuteKey = `rate_limit:${clientIP}:${Math.floor(now / 60000)}`;
  const hourKey = `rate_limit:${clientIP}:${Math.floor(now / 3600000)}`;

  try {
    // Check minute limit
    const minuteData = await kv.get(minuteKey);
    const minuteCount = minuteData ? parseInt(minuteData) : 0;
    
    if (minuteCount >= CONFIG.RATE_LIMIT.REQUESTS_PER_MINUTE) {
      return {
        allowed: false,
        resetTime: Math.floor(now / 60000) * 60 + 60,
      };
    }

    // Check hour limit
    const hourData = await kv.get(hourKey);
    const hourCount = hourData ? parseInt(hourData) : 0;
    
    if (hourCount >= CONFIG.RATE_LIMIT.REQUESTS_PER_HOUR) {
      return {
        allowed: false,
        resetTime: Math.floor(now / 3600000) * 3600 + 3600,
      };
    }

    return { allowed: true };
  } catch (error) {
    console.error('Rate limit check failed:', error);
    // Allow request if rate limit check fails
    return { allowed: true };
  }
}

/**
 * Update rate limit counters
 */
async function updateRateLimit(clientIP: string, kv: KVNamespace): Promise<void> {
  const now = Date.now();
  const minuteKey = `rate_limit:${clientIP}:${Math.floor(now / 60000)}`;
  const hourKey = `rate_limit:${clientIP}:${Math.floor(now / 3600000)}`;

  try {
    // Update minute counter
    const minuteData = await kv.get(minuteKey);
    const minuteCount = minuteData ? parseInt(minuteData) + 1 : 1;
    await kv.put(minuteKey, minuteCount.toString(), { expirationTtl: 120 });

    // Update hour counter
    const hourData = await kv.get(hourKey);
    const hourCount = hourData ? parseInt(hourData) + 1 : 1;
    await kv.put(hourKey, hourCount.toString(), { expirationTtl: 7200 });
  } catch (error) {
    console.error('Rate limit update failed:', error);
  }
}

/**
 * Generate cache key for request
 */
async function generateCacheKey(request: Request, path: string): Promise<string> {
  const url = new URL(request.url);
  const searchParams = url.searchParams.toString();
  
  let body = '';
  if (request.method === 'POST') {
    try {
      const requestClone = request.clone();
      body = await requestClone.text();
    } catch (error) {
      console.error('Failed to read request body for cache key:', error);
    }
  }

  const keyData = `${request.method}:${path}:${searchParams}:${body}`;
  
  // Create hash of the key data
  const encoder = new TextEncoder();
  const data = encoder.encode(keyData);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  
  return `cache:${hashHex}`;
}

/**
 * Get cached item
 */
async function getCachedItem(kv: KVNamespace, key: string): Promise<CacheItem | null> {
  try {
    const cached = await kv.get(key);
    if (!cached) return null;

    const item: CacheItem = JSON.parse(cached);
    const now = Date.now();
    
    // Check if cache item is still valid
    if (now - item.timestamp > item.ttl * 1000) {
      // Cache expired, delete it
      await kv.delete(key);
      return null;
    }

    return item;
  } catch (error) {
    console.error('Cache get failed:', error);
    return null;
  }
}

/**
 * Set cached item
 */
async function setCachedItem(
  kv: KVNamespace,
  key: string,
  data: any,
  ttlSeconds: number
): Promise<void> {
  try {
    const item: CacheItem = {
      data,
      timestamp: Date.now(),
      ttl: ttlSeconds,
    };

    await kv.put(key, JSON.stringify(item), {
      expirationTtl: ttlSeconds + 60, // Add buffer to KV TTL
    });
  } catch (error) {
    console.error('Cache set failed:', error);
  }
}

/**
 * Create success response
 */
function createSuccessResponse(data: any, additionalHeaders: Record<string, string> = {}): Response {
  const headers = {
    'Content-Type': 'application/json',
    ...CONFIG.CORS_HEADERS,
    ...additionalHeaders,
  };

  return new Response(JSON.stringify(data), {
    status: 200,
    headers,
  });
}

/**
 * Create error response
 */
function createErrorResponse(
  status: number,
  message: string,
  additionalHeaders: Record<string, string> = {}
): Response {
  const headers = {
    'Content-Type': 'application/json',
    ...CONFIG.CORS_HEADERS,
    ...additionalHeaders,
  };

  const errorData = {
    error: message,
    status,
    timestamp: new Date().toISOString(),
  };

  return new Response(JSON.stringify(errorData), {
    status,
    headers,
  });
}

/**
 * Utility function to validate prediction request
 */
function validatePredictionRequest(data: any): data is PredictionRequest {
  return (
    typeof data === 'object' &&
    data !== null &&
    typeof data.currency === 'string' &&
    data.currency.length > 0 &&
    (data.prediction_horizon === undefined || typeof data.prediction_horizon === 'string')
  );
}

/**
 * Utility function to validate batch prediction request
 */
function validateBatchPredictionRequest(data: any): data is BatchPredictionRequest {
  return (
    typeof data === 'object' &&
    data !== null &&
    Array.isArray(data.currencies) &&
    data.currencies.length > 0 &&
    data.currencies.every((currency: any) => typeof currency === 'string') &&
    (data.prediction_horizon === undefined || typeof data.prediction_horizon === 'string')
  );
} 