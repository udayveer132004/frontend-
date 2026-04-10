import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';

export default function FeatureCard({ icon: Icon, title, description, destination, highlight }) {
  const navigate = useNavigate();

  return (
    <div className={`feature-card-bento ${highlight ? 'highlight' : ''}`}>
      <div className="feature-icon">
        <Icon size={32} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <button 
        className={`feature-action ${highlight ? 'highlight-btn' : ''}`}
        onClick={() => navigate(destination)}
      >
        Get Started
        <ArrowRight size={18} />
      </button>
    </div>
  );
}
