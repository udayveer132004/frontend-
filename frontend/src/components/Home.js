import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Target, MessageCircle, Search, BarChart3, Mic, Sparkles } from 'lucide-react';
import FeatureCard from './FeatureCard';

export default function Home() {
  const navigate = useNavigate();

  const features = [
    { 
      icon: Upload, 
      title: 'Resume Upload', 
      description: 'Upload and parse your resume with AI-powered extraction', 
      destination: '/resume' 
    },
    { 
      icon: Target, 
      title: 'ATS Scorer', 
      description: 'Score your resume against job descriptions instantly', 
      destination: '/ats' 
    },
    { 
      icon: MessageCircle, 
      title: 'Chat with Resume', 
      description: 'Ask questions about your experience and skills', 
      destination: '/chat' 
    },
    { 
      icon: Search, 
      title: 'Job Search', 
      description: 'Discover opportunities matched to your profile', 
      destination: '/jobs',
      highlight: true
    },
    { 
      icon: BarChart3, 
      title: 'Application Tracker', 
      description: 'Track and manage all your job applications', 
      destination: '/dashboard' 
    },
    { 
      icon: Mic, 
      title: 'Mock Interview', 
      description: 'Practice interviews with AI-powered feedback', 
      destination: '/interview' 
    }
  ];

  return (
    <div className="home-dark">
      <div className="gradient-glow"></div>
      
      <div className="hero-dark">
        <div className="hero-badge">
          <Sparkles size={16} />
          <span>Powered by AI</span>
        </div>
        <h1 className="gradient-text">Infinite Scale</h1>
        <p className="hero-subtitle">Your AI-powered career companion for the modern job search</p>
        <button className="hero-cta" onClick={() => navigate('/resume')}>
          Upload Resume
        </button>
      </div>

      <div className="bento-grid">
        {features.map((feature, index) => (
          <FeatureCard
            key={index}
            icon={feature.icon}
            title={feature.title}
            description={feature.description}
            destination={feature.destination}
            highlight={feature.highlight}
          />
        ))}
      </div>
    </div>
  );
}
