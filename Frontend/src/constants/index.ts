export const API_ENDPOINTS = {
  ANALYZE: '/api/v1/analyze',
  HEALTH: '/api/v1/health',
  BATCH_ANALYZE: '/api/v1/analyze/batch',
} as const;

export const SEVERITY_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
} as const;

export const DETECTION_CATEGORIES = {
  HARASSMENT: 'harassment',
  THREAT: 'threat',
  HATE_SPEECH: 'hate_speech',
  SEXUAL_HARASSMENT: 'sexual_harassment',
  DOXXING: 'doxxing',
  EXCLUSION: 'exclusion',
} as const;

export const SUPPORTED_LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'nl', name: 'Dutch' },
  { code: 'ru', name: 'Russian' },
] as const;

export const DEFAULT_CONFIG = {
  API_TIMEOUT: 10000,
  MAX_MESSAGE_LENGTH: 1000,
  DETECTION_THRESHOLD: 0.7,
  DEBOUNCE_DELAY: 300,
} as const;