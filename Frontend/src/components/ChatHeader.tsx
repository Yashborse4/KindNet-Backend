import React from 'react';
import { MoreVertical, Shield, AlertTriangle, Search } from 'lucide-react';

interface ChatHeaderProps {
  title?: string;
  subtitle?: string;
  isOnline?: boolean;
  avatar?: string;
  showControls?: boolean;
  detectionEnabled?: boolean;
  onToggleDetection?: () => void;
  backendStatus?: 'online' | 'offline' | 'checking';
  lastBackendCheck?: Date;
}

const ChatHeader: React.FC<ChatHeaderProps> = ({
  title = "AI Safety Chat",
  subtitle = "Cyberbullying Detection System",
  isOnline = true,
  avatar,
  showControls = true,
  detectionEnabled = true,
  onToggleDetection,
  backendStatus,
  lastBackendCheck
}) => {
  return (
    <div className="p-4 bg-black/40 border-b border-white/10 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="relative">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-white to-neutral-400 flex items-center justify-center shadow-lg shadow-black/50">
            <Shield size={24} className="text-black" />
          </div>
          {isOnline && (
            <div className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full bg-emerald-400 border-2 border-black"></div>
          )}
        </div>
        <div>
          <h1 className="text-white font-semibold text-base md:text-lg tracking-tight">{title}</h1>
          <p className="text-white/60 text-[11px] md:text-xs uppercase tracking-[0.18em]">{subtitle}</p>
        </div>
      </div>

      {showControls && (
        <div className="flex items-center gap-3">
          {/* Backend status pill */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[11px] text-white/70">
            <span
              className={`inline-flex w-2 h-2 rounded-full ${
                backendStatus === 'online'
                  ? 'bg-emerald-400'
                  : backendStatus === 'offline'
                  ? 'bg-red-400'
                  : 'bg-zinc-400 animate-pulse'
              }`}
            />
            <span>
              {backendStatus === 'online'
                ? 'Backend online'
                : backendStatus === 'offline'
                ? 'Backend offline'
                : 'Checking backend...'}
            </span>
            {lastBackendCheck && backendStatus !== 'checking' && (
              <span className="text-white/40">
                · {lastBackendCheck.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
          </div>

          <button
            onClick={onToggleDetection}
            className={`btn-icon ${detectionEnabled ? 'bg-white text-black border-white/60' : 'bg-black border-white/40 text-white/80'}`}
            title={detectionEnabled ? 'AI Protection Active' : 'AI Protection Disabled'}
          >
            {detectionEnabled ? <Shield size={16} /> : <AlertTriangle size={16} />}
          </button>
          <button className="btn-icon bg-white/0 border-white/20">
            <Search size={16} className="text-white/70" />
          </button>
          <button className="btn-icon bg-white/0 border-white/20">
            <MoreVertical size={16} className="text-white/70" />
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatHeader;
