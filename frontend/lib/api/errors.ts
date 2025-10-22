/**
 * Custom error classes for API operations
 */

/**
 * API error response type
 */
interface ApiErrorResponse {
  message: string;
  detail?: string;
  status?: number;
  [key: string]: unknown;
}

export class ApiRequestError extends Error {
  public readonly status: number;
  public readonly statusText: string;
  public readonly response?: ApiErrorResponse;

  constructor(
    message: string,
    status: number,
    statusText: string,
    response?: ApiErrorResponse
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.statusText = statusText;
    this.response = response;
  }
}

export class NetworkError extends Error {
  public readonly originalError?: Error;

  constructor(message: string, originalError?: Error) {
    super(message);
    this.name = "NetworkError";
    this.originalError = originalError;
  }
}

export class TimeoutError extends Error {
  public readonly timeout: number;

  constructor(message: string, timeout: number) {
    super(message);
    this.name = "TimeoutError";
    this.timeout = timeout;
  }
}

export class ValidationError extends Error {
  public readonly field?: string;
  public readonly value?: unknown;

  constructor(message: string, field?: string, value?: unknown) {
    super(message);
    this.name = "ValidationError";
    this.field = field;
    this.value = value;
  }
}

/**
 * Determines if an error is retryable
 */
export function isRetryableError(error: Error): boolean {
  if (error instanceof NetworkError) {
    return true;
  }
  
  if (error instanceof TimeoutError) {
    return true;
  }
  
  if (error instanceof ApiRequestError) {
    // Retry on 5xx errors and some 4xx errors
    return error.status >= 500 || error.status === 429;
  }
  
  return false;
}

/**
 * Calculates retry delay with exponential backoff
 */
export function getRetryDelay(attempt: number, baseDelay: number = 1000): number {
  const maxDelay = 30000; // 30 seconds
  const delay = baseDelay * Math.pow(2, attempt - 1);
  return Math.min(delay, maxDelay);
}

/**
 * Parses error response from API
 */
export function parseErrorResponse(response: Response): Promise<ApiErrorResponse> {
  const contentType = response.headers.get("content-type");
  
  if (contentType?.includes("application/json")) {
    return response.json() as Promise<ApiErrorResponse>;
  }
  
  return response.text().then(text => ({ message: text }));
}

/**
 * Creates a standardized error message
 */
export function createErrorMessage(error: Error): string {
  if (error instanceof ApiRequestError) {
    return `API Error ${error.status}: ${error.message}`;
  }
  
  if (error instanceof NetworkError) {
    return `Network Error: ${error.message}`;
  }
  
  if (error instanceof TimeoutError) {
    return `Request Timeout: ${error.message}`;
  }
  
  if (error instanceof ValidationError) {
    return `Validation Error: ${error.message}`;
  }
  
  return error.message || "An unexpected error occurred";
}

/**
 * Error type guard functions
 */
export function isApiRequestError(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError;
}

export function isNetworkError(error: unknown): error is NetworkError {
  return error instanceof NetworkError;
}

export function isTimeoutError(error: unknown): error is TimeoutError {
  return error instanceof TimeoutError;
}

export function isValidationError(error: unknown): error is ValidationError {
  return error instanceof ValidationError;
}
