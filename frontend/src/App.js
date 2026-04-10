import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Upload, Target, MessageCircle, Search, BarChart3, Mic, Home as HomeIcon } from 'lucide-react';
import Home from './components/Home';
import ResumeUpload from './components/ResumeUpload';
import ATSScore from './components/ATSScore';
import ResumeChat from './components/ResumeChat';
import JobSearch from './components/JobSearch';
import Dashboard from './components/Dashboard';
import MockInterview from './components/MockInterview';
import ChatbotMascot from './components/ChatbotMascot';
import './App.css';

function Navigation() {
  const location = useLocation();
  const isHome = location.pathname === '/';

  const navItems = [
    { path: '/', label: 'Home', icon: HomeIcon },
    { path: '/resume', label: 'Resume Upload', icon: Upload },
    { path: '/ats', label: 'ATS Score', icon: Target },
    { path: '/chat', label: 'Resume Chat', icon: MessageCircle },
    { path: '/jobs', label: 'Job Search', icon: Search },
    { path: '/dashboard', label: 'Dashboard', icon: BarChart3 },
    { path: '/interview', label: 'Mock Interview', icon: Mic }
  ];

  return (
    <nav className="navbar">
      {navItems.map(item => {
        const Icon = item.icon;
        return (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
          >
            <Icon size={18} />
            <span>{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="app">
        <header>
          <h1>🤖 AI Job Assistant</h1>
        </header>
        
        <Navigation />

        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/resume" element={<ResumeUpload />} />
            <Route path="/ats" element={<ATSScore />} />
            <Route path="/chat" element={<ResumeChat />} />
            <Route path="/jobs" element={<JobSearch />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/interview" element={<MockInterview />} />
          </Routes>
        </main>

        <ChatbotMascot />
      </div>
    </Router>
  );
}

export default App;
