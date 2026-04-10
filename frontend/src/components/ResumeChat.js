import React, { useState } from 'react';
import { MessageCircle, Send } from 'lucide-react';
import { mockApi } from '../services/mockApi';

export default function ResumeChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    const response = await mockApi.chatWithResume(input);
    setMessages(prev => [...prev, { role: 'assistant', content: response }]);
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <h2><MessageCircle size={24} /> Resume Chat</h2>
      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">Ask questions about your resume...</div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-content">{msg.content}</div>
            </div>
          ))}
          {loading && <div className="message assistant"><div className="message-content">Thinking...</div></div>}
        </div>
        <div className="chat-input">
          <input
            type="text"
            placeholder="Ask about your experience, skills, etc..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          />
          <button onClick={sendMessage}><Send size={20} /></button>
        </div>
      </div>
    </div>
  );
}
