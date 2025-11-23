import React from 'react';
import { User, Bot } from 'lucide-react';

interface MessageProps {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  avatar?: string;
  username?: string;
  isFlagged?: boolean;
  confidence?: number;
  severity?: string;
}

const Message: React.FC<MessageProps> = ({ 
  text, 
  isUser, 
  timestamp, 
  avatar, 
  username,
  isFlagged,
  confidence,
  severity 
}) => {
  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    }).format(date);
  };

  return (
    <div className={`flex items-start gap-3 animate-fade-in ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center ${isUser ? 'bg-white' : 'bg-gradient-to-br from-gray-600 to-gray-800'}`}>
        {avatar ? (
          <img src={avatar} alt={username} className="w-full h-full rounded-full object-cover" />
        ) : isUser ? (
          <User size={20} className="text-black" />
        ) : (
          <Bot size={20} className="text-white" />
        )}
      </div>
      <div className={`flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
        <div className={`flex items-center gap-2 text-xs text-white/50 ${isUser ? 'flex-row-reverse' : ''}`}>
          <span>{username ? (isUser ? 'You' : username) : (isUser ? 'You' : 'Bot')}</span>
          <span>&bull;</span>
          <span>{formatTime(timestamp)}</span>
        </div>
        <div
          className={`card-dark px-4 py-2.5 rounded-2xl max-w-md ${isUser
              ? 'rounded-br-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white'
              : 'rounded-bl-lg'
            }`}>
          <p className="text-sm leading-relaxed">{text}</p>

          {/* Bullying risk indicator for assistant messages with analysis */}
          {!isUser && typeof confidence === 'number' && (
            <div className="mt-2 flex flex-col gap-1 text-[11px] text-white/60">
              <div className="flex items-center gap-2">
                {typeof severity === 'string' && severity !== 'none' && (
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                      isFlagged
                        ? 'bg-red-500/15 border-red-500/50 text-red-200'
                        : 'bg-emerald-500/15 border-emerald-500/50 text-emerald-200'
                    }`}
                  >
                    {severity.toUpperCase()}
                  </span>
                )}
                <span>
                  {Math.round((confidence ?? 0) * 100)}% bullying risk
                </span>
              </div>
              <div className="w-32 h-1.5 rounded-full bg-white/10 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    isFlagged
                      ? 'bg-gradient-to-r from-red-500 to-orange-500'
                      : 'bg-gradient-to-r from-emerald-400 to-sky-400'
                  }`}
                  style={{
                    width: `${Math.max(
                      0,
                      Math.min(100, Math.round((confidence ?? 0) * 100))
                    )}%`,
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;
