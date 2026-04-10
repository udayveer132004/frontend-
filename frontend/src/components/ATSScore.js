import React, { useState } from 'react';
import { Target, CheckCircle, XCircle } from 'lucide-react';
import { mockApi } from '../services/mockApi';

export default function ATSScore() {
  const [jobDesc, setJobDesc] = useState('');
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeScore = async () => {
    if (!jobDesc.trim()) return;
    setLoading(true);
    const result = await mockApi.calculateATSScore(null, jobDesc);
    setScore(result);
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <h2>ATS Score Analysis</h2>
      <textarea
        placeholder="Paste job description here..."
        value={jobDesc}
        onChange={(e) => setJobDesc(e.target.value)}
        rows={8}
      />
      <button onClick={analyzeScore} disabled={loading}>
        {loading ? 'Analyzing...' : 'Calculate ATS Score'}
      </button>

      {score && (
        <div className="score-result">
          <div className="score-circle">
            <Target size={40} />
            <h1>{score.score}%</h1>
            <p>ATS Match Score</p>
          </div>

          <div className="keywords-section">
            <div className="matched">
              <h4><CheckCircle size={18} /> Matched Keywords</h4>
              {score.matchedKeywords.map(k => <span key={k} className="keyword matched">{k}</span>)}
            </div>
            <div className="missing">
              <h4><XCircle size={18} /> Missing Keywords</h4>
              {score.missingKeywords.map(k => <span key={k} className="keyword missing">{k}</span>)}
            </div>
          </div>

          <div className="suggestions">
            <h4>Suggestions</h4>
            <ul>{score.suggestions.map((s, i) => <li key={i}>{s}</li>)}</ul>
          </div>
        </div>
      )}
    </div>
  );
}
