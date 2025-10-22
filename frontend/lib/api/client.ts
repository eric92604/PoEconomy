/**
 * Base API client with error handling and retries
 */

import { API_BASE_URL, API_KEY, API_TIMEOUT, MAX_RETRIES } from "@/lib/constants/api";
import {
  ApiRequestError,
  NetworkError,
  getRetryDelay,
  isRetryableError,
  parseErrorResponse,
} from "./errors";

/**
 * Request options
 */
interface RequestOptions extends RequestInit {
  timeout?: number;
  retries?: number;
}

/**
 * Create an AbortController with timeout
 */
function createTimeoutSignal(timeout: number): AbortSignal {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), timeout);
  return controller.signal;
}

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Make an API request with retries and error handling
 */
async function makeRequest<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const {
    timeout = API_TIMEOUT,
    retries = MAX_RETRIES,
    headers = {},
    ...fetchOptions
  } = options;

  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders: HeadersInit = {
    "Content-Type": "application/json",
    ...(API_KEY && { "x-api-key": API_KEY }),
    ...headers,
  };

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const signal = createTimeoutSignal(timeout);

      const response = await fetch(url, {
        ...fetchOptions,
        headers: defaultHeaders,
        signal,
      });

      // Handle non-OK responses
      if (!response.ok) {
        const errorResponse = await parseErrorResponse(response);
        const error = new ApiRequestError(
          errorResponse.message || `HTTP ${response.status}`,
          response.status,
          response.statusText,
          errorResponse
        );

        // Retry on retryable errors
        if (attempt < retries && isRetryableError(error)) {
          const delay = getRetryDelay(attempt);
          await sleep(delay);
          continue;
        }

        throw error;
      }

      // Parse and return response
      const data = await response.json();
      return data as T;
    } catch (error) {
      lastError = error as Error;

      // Handle network errors
      if (error instanceof TypeError || (error as Error).name === "AbortError") {
        const networkError = new NetworkError(
          error instanceof TypeError ? "Network request failed" : "Request timeout"
        );

        // Retry network errors
        if (attempt < retries) {
          const delay = getRetryDelay(attempt);
          await sleep(delay);
          continue;
        }

        throw networkError;
      }

      // Handle API errors
      if (error instanceof ApiRequestError) {
        // Retry retryable errors
        if (attempt < retries && isRetryableError(error)) {
          const delay = getRetryDelay(attempt);
          await sleep(delay);
          continue;
        }

        throw error;
      }

      // Unknown error, don't retry
      throw error;
    }
  }

  // If all retries failed, throw the last error
  throw lastError || new Error("Request failed after all retries");
}

/**
 * API client
 */
export const apiClient = {
  /**
   * Make a GET request
   */
  get: <T>(endpoint: string, options?: RequestOptions): Promise<T> => {
    return makeRequest<T>(endpoint, {
      ...options,
      method: "GET",
    });
  },

  /**
   * Make a POST request
   */
  post: <T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<T> => {
    return makeRequest<T>(endpoint, {
      ...options,
      method: "POST",
      body: data ? JSON.stringify(data) : undefined,
    });
  },

  /**
   * Make a PUT request
   */
  put: <T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<T> => {
    return makeRequest<T>(endpoint, {
      ...options,
      method: "PUT",
      body: data ? JSON.stringify(data) : undefined,
    });
  },

  /**
   * Make a DELETE request
   */
  delete: <T>(endpoint: string, options?: RequestOptions): Promise<T> => {
    return makeRequest<T>(endpoint, {
      ...options,
      method: "DELETE",
    });
  },
};



