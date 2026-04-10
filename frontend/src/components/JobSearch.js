import React, { useState, useEffect } from 'react';
import { Search, MapPin, DollarSign, TrendingUp, ExternalLink } from 'lucide-react';
import { mockApi } from '../services/mockApi';

export default function JobSearch() {
  const [query, setQuery] = useState('software developer');
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    searchJobs();
  }, []);

  const searchJobs = async () => {
    setLoading(true);
    const results = await mockApi.searchJobs(query);
    setJobs(results);
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <h2>Job Search</h2>
      <div className="search-bar">
        <Search size={20} />
        <input
          type="text"
          placeholder="Search jobs by title, skills, or company..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && searchJobs()}
        />
        <button onClick={searchJobs}>Search</button>
      </div>

      {loading && <div className="loading">Searching jobs...</div>}

      <div className="jobs-list">
        {jobs.map(job => (
          <div key={job.id} className="job-card">
            <div className="job-header">
              <h3>{job.title}</h3>
              <span className="match-score">{job.match}% Match</span>
            </div>
            <p className="company">{job.company}</p>
            <div className="job-details">
              <span><MapPin size={16} /> {job.location}</span>
              <span><DollarSign size={16} /> {job.salary}</span>
              <span><TrendingUp size={16} /> {job.posted}</span>
            </div>
            {job.description && (
              <p className="job-description">{job.description.substring(0, 150)}...</p>
            )}
            <div className="job-actions">
              <button className="apply-btn">Apply Now</button>
              {job.link && (
                <a href={job.link} target="_blank" rel="noopener noreferrer" className="view-link">
                  <ExternalLink size={16} /> View Details
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
