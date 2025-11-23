import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import MessageInput from './MessageInput';
import ChatHeader from './ChatHeader';
import { api } from '../services/api';

interface MessageType {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  username?: string;
  avatar?: string;
  isFlagged?: boolean;
  confidence?: number;
  severity?: string;
  riskIndicators?: string[];
}

interface ChatAppProps {
  onSendMessage?: (message: string) => Promise<void>;
  className?: string;
}

const ChatApp: React.FC<ChatAppProps> = ({ onSendMessage, className = "" }) => {
  const [messages, setMessages] = useState<MessageType[]>([
    {
      id: '1',
      text: 'Welcome to AI Safety Chat! I\'m here to help ensure respectful communication by analyzing messages for harmful content in real-time.',
      isUser: false,
      timestamp: new Date(Date.now() - 60000),
      username: 'AI Assistant'
    },
    {
      id: '2',
      text: 'Try sending a message - I\'ll analyze it for cyberbullying, harassment, or other inappropriate content and provide feedback.',
      isUser: false,
      timestamp: new Date(Date.now() - 30000),
      username: 'AI Assistant'
    }
  ]);
  
  const [isTyping, setIsTyping] = useState(false);
  const [detectionEnabled, setDetectionEnabled] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean | null>(null);
  const [lastBackendCheck, setLastBackendCheck] = useState<Date | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Backend availability check every 20 seconds
  useEffect(() => {
    let isMounted = true;

    const checkBackend = async () => {
      try {
        const available = await api.checkAvailability();
        if (!isMounted) return;
        setIsBackendOnline(available);
        setLastBackendCheck(new Date());
      } catch (error) {
        console.error('Backend health check failed:', error);
        if (!isMounted) return;
        setIsBackendOnline(false);
        setLastBackendCheck(new Date());
      }
    };

    // Initial check immediately
    checkBackend();

    // Poll every 20 seconds
    const intervalId = setInterval(checkBackend, 20000);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  // API call for cyberbullying detection
  type AnalysisResult = {
    isFlagged: boolean;
    confidence: number;
    severity?: string;
    riskIndicators?: string[];
    error?: string;
  };

  const analyzeMessage = async (text: string): Promise<AnalysisResult> => {
    try {
      const { result, error } = await api.quickDetect(text, 0.7);
      
      if (error) {
        console.error('Detection API error:', error);
        return {
          isFlagged: false,
          confidence: 0,
          error
        };
      }
      
      if (result) {
        return {
          isFlagged: result.is_bullying,
          confidence: result.confidence,
          severity: result.severity,
          riskIndicators: result.risk_indicators,
        };
      }
      
      // Fallback if no result
      return {
        isFlagged: false,
        confidence: 0
      };
    } catch (error) {
      console.error('Unexpected detection error:', error);
      return {
        isFlagged: false,
        confidence: 0,
        error: 'Detection service unavailable'
      };
    }
  };

  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMessage: MessageType = {
      id: Date.now().toString(),
      text,
      isUser: true,
      timestamp: new Date(),
      username: 'You'
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Analyze message for cyberbullying if detection is enabled
      let analysisResult: AnalysisResult = { isFlagged: false, confidence: 0 };
      if (detectionEnabled) {
        analysisResult = await analyzeMessage(text);
      }

      // Call external API if provided
      if (onSendMessage) {
        await onSendMessage(text);
      }

      // Show typing indicator
      setIsTyping(true);
      
      // Process response after a brief delay
      setTimeout(() => {
        setIsTyping(false);

        // Generate response based on analysis
        let responseText = '';
        let responseIcon = '';
        
        if (analysisResult.error) {
          responseIcon = '🔧';
          responseText = `Service temporarily unavailable: ${analysisResult.error}. Your message has been received but not analyzed for safety.`;
        } else if (detectionEnabled && analysisResult.isFlagged) {
          responseIcon = '⚠️';
          responseText = `Warning: This message was flagged as potentially harmful (${Math.round(analysisResult.confidence * 100)}% confidence). Please be respectful in your communication.`;
        } else if (detectionEnabled) {
          responseIcon = '✅';
          responseText = 'Message analyzed and cleared. No harmful content detected.';
        } else {
          responseIcon = '💬';
          responseText = 'Message received. Content analysis is currently disabled.';
        }

        const botResponse: MessageType = {
          id: (Date.now() + 1).toString(),
          text: `${responseIcon} ${responseText}`,
          isUser: false,
          timestamp: new Date(),
          username: 'AI Safety Assistant',
          isFlagged: analysisResult.isFlagged,
          confidence: analysisResult.confidence,
          severity: analysisResult.severity,
          riskIndicators: analysisResult.riskIndicators,
        };

        setMessages(prev => [...prev, botResponse]);
      }, 1000);

    } catch (error) {
      console.error('Error sending message:', error);
      setIsTyping(false);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleDetection = () => {
    setDetectionEnabled(!detectionEnabled);
    
    const statusMessage: MessageType = {
      id: Date.now().toString(),
      text: `🔄 Cyberbullying detection has been ${!detectionEnabled ? 'enabled' : 'disabled'}.`,
      isUser: false,
      timestamp: new Date(),
      username: 'System'
    };
    
    setMessages(prev => [...prev, statusMessage]);
  };

  const backendStatus: 'online' | 'offline' | 'checking' =
    isBackendOnline === null ? 'checking' : isBackendOnline ? 'online' : 'offline';

  return (
    <div className={`flex flex-col h-full max-h-full bg-transparent ${className}`}>
      {/* Chat Header */}
      <ChatHeader 
        detectionEnabled={detectionEnabled}
        onToggleDetection={toggleDetection}
        backendStatus={backendStatus}
        lastBackendCheck={lastBackendCheck ?? undefined}
      />

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto scrollbar-modern p-6 space-y-6">
        {messages.map((message) => (
          <Message
            key={message.id}
            id={message.id}
            text={message.text}
            isUser={message.isUser}
            timestamp={message.timestamp}
            username={message.username}
            avatar={message.avatar}
            isFlagged={message.isFlagged}
            confidence={message.confidence}
            severity={message.severity}
          />
        ))}
        
        {/* Typing Indicator */}
        {(isTyping || isLoading) && (
          <div className="flex items-start gap-4 p-6 fade-in">
            <div className="flex-shrink-0 w-11 h-11 rounded-xl bg-black/40 border border-white/20 text-white backdrop-blur-xl flex items-center justify-center">
              <div className="w-3 h-3 bg-white/30 rounded-full animate-pulse"></div>
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-medium text-white/80">AI Assistant</span>
                <span className="text-xs text-white/40">
                  {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
              <div className="chat-bubble chat-bubble-other scale-in">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
                    <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
                  </div>
                  <span className="text-xs text-white/70 font-medium">Analyzing...</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Message Input */}
      <MessageInput
        onSendMessage={handleSendMessage}
        disabled={isLoading}
        placeholder={detectionEnabled ? "Type your message... (AI monitoring active)" : "Type your message... (monitoring disabled)"}
      />
    </div>
  );
};

export default ChatApp;
