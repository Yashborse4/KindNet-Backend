import {
  ApiResponse,
  DetectionResult,
  BatchDetectionResponse,
  DetectionStats,
  DetectRequest,
  BatchDetectRequest,
  AddWordsRequest,
  AddWordsResponse,
  ApiError,
  ApiConfig
} from '../types/api';

class CyberBullyingAPI {
  private config: ApiConfig;

  constructor(config?: Partial<ApiConfig>) {
    this.config = {
      baseUrl: process.env.REACT_APP_API_URL || 'http://localhost:5000',
      timeout: 30000, // 30 seconds
      retries: 3,
      ...config
    };
  }

  /**
   * Generic fetch wrapper with error handling and retries
   */
  private async fetchWithRetry<T>(
    endpoint: string,
    options: RequestInit = {},
    attempt: number = 1
  ): Promise<ApiResponse<T>> {
    const url = `${this.config.baseUrl}${endpoint}`;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new ApiError(`HTTP ${response.status}: ${response.statusText}`, response.status, new Date().toISOString());
      }

      const data: ApiResponse<T> = await response.json();
      
      if (!data.success && data.error) {
        throw new ApiError(data.error, response.status, data.timestamp);
      }

      return data;

    } catch (error) {
      console.error(`API Error (attempt ${attempt}):`, error);

      // Retry logic for network errors and server errors (5xx)
      if (attempt < this.config.retries && this.shouldRetry(error)) {
        console.log(`Retrying request to ${endpoint} (attempt ${attempt + 1}/${this.config.retries})`);
        await this.delay(Math.pow(2, attempt) * 1000); // Exponential backoff
        return this.fetchWithRetry<T>(endpoint, options, attempt + 1);
      }

      // Convert error to ApiError if it isn't already
      if (error instanceof ApiError) {
        throw error;
      } else if (error instanceof DOMException && error.name === 'AbortError') {
        throw new ApiError('Request timeout', 408, new Date().toISOString());
      } else {
        throw new ApiError(
          error instanceof Error ? error.message : 'Network error',
          0,
          new Date().toISOString()
        );
      }
    }
  }

  /**
   * Determine if an error should trigger a retry
   */
  private shouldRetry(error: any): boolean {
    // Retry for network errors, timeouts, and 5xx server errors
    if (error instanceof ApiError) {
      return error.status >= 500 || error.status === 0 || error.status === 408;
    }
    return error instanceof DOMException && error.name === 'AbortError';
  }

  /**
   * Add delay for retry logic
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<ApiResponse> {
    return this.fetchWithRetry('/');
  }

  /**
   * Detect cyberbullying in a single message
   */
  async detectBullying(request: DetectRequest): Promise<ApiResponse<DetectionResult>> {
    return this.fetchWithRetry<DetectionResult>('/api/detect', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Detect cyberbullying in multiple messages
   */
  async batchDetectBullying(request: BatchDetectRequest): Promise<ApiResponse<BatchDetectionResponse>> {
    return this.fetchWithRetry<BatchDetectionResponse>('/api/batch-detect', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get detection statistics
   */
  async getStats(): Promise<ApiResponse<DetectionStats>> {
    return this.fetchWithRetry<DetectionStats>('/api/stats');
  }

  /**
   * Add new bullying words to the database
   */
  async addBullyingWords(request: AddWordsRequest): Promise<ApiResponse<AddWordsResponse>> {
    return this.fetchWithRetry<AddWordsResponse>('/api/add-words', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Quick detection method for chat integration
   * Returns only the result with simplified error handling
   */
  async quickDetect(
    text: string, 
    confidenceThreshold: number = 0.7
  ): Promise<{ result: DetectionResult | null; error: string | null }> {
    try {
      const response = await this.detectBullying({
        text,
        confidence_threshold: confidenceThreshold,
        include_details: true
      });

      return {
        result: response.data || null,
        error: null
      };
    } catch (error) {
      console.error('Quick detect error:', error);
      return {
        result: null,
        error: error instanceof ApiError ? error.message : 'Detection failed'
      };
    }
  }

  /**
   * Check if the API is available
   */
  async checkAvailability(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Update API configuration
   */
  updateConfig(config: Partial<ApiConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): ApiConfig {
    return { ...this.config };
  }
}

// Create and export a default instance
export const api = new CyberBullyingAPI();

// Export the class for custom instances
export default CyberBullyingAPI;

// Export the ApiError class from types
export { ApiError } from '../types/api';
