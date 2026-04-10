import React, { useState } from 'react';

export default function ChatbotMascot() {
  const [message, setMessage] = useState("Hi! I'm your AI assistant!");

  const messages = [
    "Hi! I'm your AI assistant!",
    "Need help with your job search?",
    "Let's find your dream job!",
    "Upload your resume to get started!",
    "I'm here to help 24/7!"
  ];

  const handleClick = () => {
    const randomMessage = messages[Math.floor(Math.random() * messages.length)];
    setMessage(randomMessage);
  };

  return (
    <div className="chatbot-mascot" onClick={handleClick}>
      <div className="mascot-tooltip">{message}</div>
      <div className="mascot-body">
        <div className="mascot-head">
          <div className="mascot-antenna"></div>
          <div className="mascot-eyes">
            <div className="mascot-eye"></div>
            <div className="mascot-eye"></div>
          </div>
          <div className="mascot-mouth"></div>
        </div>
        <div className="mascot-body-base"></div>
        <div className="mascot-arms">
          <div className="mascot-arm left"></div>
          <div className="mascot-arm right"></div>
        </div>
        <div className="mascot-sparkles">
          <div className="sparkle"></div>
          <div className="sparkle"></div>
          <div className="sparkle"></div>
        </div>
      </div>
    </div>
  );
}
