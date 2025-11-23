import React from 'react';
import ChatApp from './components/ChatApp';
import './index.css';

function App() {
  // You can implement your API integration here
  const handleSendMessage = async (message: string): Promise<void> => {
    // Example API call - replace with your actual endpoint
    try {
      console.log('Sending message to API:', message);
      // Add your API integration here if needed
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  return (
    <div className="App min-h-screen bg-black text-white relative overflow-hidden">
      {/* Subtle background pattern */}
      <div className="fixed inset-0">
        <div
          className="absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='white' fill-opacity='1'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }}
        ></div>
        {/* Gradient overlay - subtle black/white depth */}
        <div className="absolute inset-0 bg-gradient-to-br from-black via-zinc-900/90 to-neutral-950"></div>
      </div>

      {/* Main chat container */}
      <div className="relative z-10 flex items-center justify-center px-4 py-6 md:px-8">
        <div className="w-full max-w-5xl">
          <div className="glass-dark rounded-3xl overflow-hidden shadow-2xl border border-white/10 h-[85vh]">
            <ChatApp onSendMessage={handleSendMessage} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
