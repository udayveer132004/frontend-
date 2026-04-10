import React, { useState } from 'react';
import { Upload, FileText } from 'lucide-react';
import { mockApi } from '../services/mockApi';

export default function ResumeUpload() {
  const [parsedData, setParsedData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setLoading(true);
    const data = await mockApi.parseResume(file);
    setParsedData(data);
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <h2>Resume Upload</h2>
      <div className="upload-area">
        <input type="file" id="resume" accept=".pdf,.doc,.docx" onChange={handleFileUpload} style={{display: 'none'}} />
        <label htmlFor="resume" className="upload-label">
          <Upload size={48} />
          <p>Click to upload resume (PDF, DOC, DOCX)</p>
        </label>
      </div>

      {loading && <div className="loading">Parsing resume...</div>}

      {parsedData && (
        <div className="parsed-resume">
          <h3><FileText size={20} /> Parsed Resume</h3>
          <div className="info-section">
            <p><strong>Name:</strong> {parsedData.name}</p>
            <p><strong>Email:</strong> {parsedData.email}</p>
            <p><strong>Phone:</strong> {parsedData.phone}</p>
          </div>
          <div className="info-section">
            <strong>Skills:</strong>
            <div className="skills">{parsedData.skills.map(s => <span key={s} className="skill-tag">{s}</span>)}</div>
          </div>
          <div className="info-section">
            <strong>Experience:</strong>
            {parsedData.experience.map((exp, i) => (
              <div key={i} className="experience-item">
                <p>{exp.title} at {exp.company}</p>
                <p className="duration">{exp.duration}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
