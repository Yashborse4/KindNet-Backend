// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

// Detection Result Types
export interface DetectedCategory {
  category: string;
  items: string[];
  score: number;
  severity: string;
}

export interface DetectionResult {
  is_bullying: boolean;
  confidence: number;
  // Enhanced detector fields
  severity?: string;
  detected_categories?: DetectedCategory[];
  risk_indicators?: string[];
  sentiment_analysis?: Record<string, number>;
  context_analysis?: {
    score: number;
    indicators: string[];
    detailed_analysis?: Record<string, any>;
  };
  intent_classification?: Record<string, number>;
  detection_method?: string;
  detected_languages?: string[];
  detected_items?: any[];
  // Legacy/compatibility fields used in some tools/tests
  flagged_words?: string[];
  explanation?: string;
  openai_used?: boolean;
  local_match?: boolean;
  processing_time?: number;
}

// Batch Detection Types
export interface BatchDetectionResult extends DetectionResult {
  index: number;
  error?: string;
}

export interface BatchDetectionResponse {
  results: BatchDetectionResult[];
  total_processed: number;
  bullying_detected: number;
}

// Statistics Types
export interface DetectionStats {
  total_requests: number;
  bullying_detected: number;
  average_confidence: number;
  openai_requests: number;
  local_matches: number;
  most_common_words: Array<{ word: string; count: number }>;
  daily_stats?: Array<{ date: string; count: number }>;
}

// Request Types
export interface DetectRequest {
  text: string;
  confidence_threshold?: number;
  include_details?: boolean;
}

export interface BatchDetectRequest {
  texts: string[];
  confidence_threshold?: number;
  include_details?: boolean;
}

export interface AddWordsRequest {
  words: string[];
}

export interface AddWordsResponse {
  words_added: number;
  total_words: number;
}

// Error Types
export interface ApiErrorInterface {
  message: string;
  status: number;
  timestamp: string;
}

// ApiError class for proper error handling
export class ApiError extends Error {
  status: number;
  timestamp: string;

  constructor(message: string, status: number, timestamp: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.timestamp = timestamp;
  }
}

// Configuration Types
export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
}

// Chat Message Types for Integration
export interface ChatMessage {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  username?: string;
  avatar?: string;
  detectionResult?: DetectionResult;
  isProcessing?: boolean;
  hasError?: boolean;
  errorMessage?: string;
}

// WebSocket Event Types (for future real-time features)
export interface WSMessage {
  type: 'detection' | 'status' | 'error';
  data: any;
  timestamp: string;
}

export interface DetectionEvent extends WSMessage {
  type: 'detection';
  data: {
    messageId: string;
    result: DetectionResult;
  };
}

export interface StatusEvent extends WSMessage {
  type: 'status';
  data: {
    connected: boolean;
    activeUsers?: number;
    serverStatus?: string;
  };
}

export interface ErrorEvent extends WSMessage {
  type: 'error';
  data: {
    code: string;
    message: string;
  };
}
