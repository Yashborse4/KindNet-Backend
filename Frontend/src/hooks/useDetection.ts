import { useState, useCallback } from 'react';
import { DetectionResult, AnalysisRequest } from '../types/api';
import { analyzeText } from '../services/api';

export interface UseDetectionReturn {
  result: DetectionResult | null;
  isLoading: boolean;
  error: string | null;
  analyze: (request: AnalysisRequest) => Promise<void>;
  clearResult: () => void;
}

export const useDetection = (): UseDetectionReturn => {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (request: AnalysisRequest) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const detectionResult = await analyzeText(request);
      setResult(detectionResult);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMessage);
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearResult = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return {
    result,
    isLoading,
    error,
    analyze,
    clearResult,
  };
};