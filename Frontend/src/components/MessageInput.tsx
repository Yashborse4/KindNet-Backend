import React, { useState, useRef, useEffect } from 'react';
import { Send, Plus, Smile, Mic, MicOff } from 'lucide-react';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

const MessageInput: React.FC<MessageInputProps> = ({ 
  onSendMessage, 
  disabled = false, 
  placeholder = "Type your message..." 
}) => {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    // Here you would implement voice recording logic
  };

  return (
    <div className="p-6 bg-black/30 border-t border-white/10">
      {isRecording && (
        <div className="flex items-center justify-center mb-3 animate-pulse">
          <div className="flex items-center gap-2 card-dark px-4 py-2 rounded-full">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-white/90 text-sm">Recording...</span>
          </div>
        </div>
      )}
      <form onSubmit={handleSubmit} className="flex items-end gap-4">
        <button type="button" className="btn-icon" disabled={disabled}>
          <Plus size={20} className="text-white/70" />
        </button>

        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="input-modern pr-12 resize-none scrollbar-hide"
            style={{ minHeight: '48px' }}
          />
          <button
            type="button"
            className="absolute right-3 top-1/2 -translate-y-1/2 btn-icon w-8 h-8"
            disabled={disabled}
          >
            <Smile size={18} className="text-white/70" />
          </button>
        </div>

        {message.trim() ? (
          <button
            type="submit"
            disabled={disabled || !message.trim()}
            className="btn-primary !px-3 !py-3 !rounded-full"
          >
            <Send size={20} />
          </button>
        ) : (
          <button
            type="button"
            onClick={toggleRecording}
            disabled={disabled}
            className={`btn-icon ${isRecording ? 'bg-red-500/50 border-red-500/80' : ''}`}
          >
            {isRecording ? <MicOff size={20} /> : <Mic size={20} className="text-white/70" />}
          </button>
        )}
      </form>
    </div>
  );
};

export default MessageInput;
