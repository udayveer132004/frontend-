import React, { useState, useEffect } from 'react';
import { BarChart3, Briefcase } from 'lucide-react';
import { mockApi } from '../services/mockApi';

export default function Dashboard() {
  const [applications, setApplications] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadApplications();
  }, []);

  const loadApplications = async () => {
    const data = await mockApi.getApplications();
    setApplications(data);
    setLoading(false);
  };

  const getStatusColor = (status) => {
    const colors = {
      'Applied': '#3b82f6',
      'Interview': '#10b981',
      'Rejected': '#ef4444',
      'Offer': '#8b5cf6'
    };
    return colors[status] || '#6b7280';
  };

  const stats = {
    total: applications.length,
    interview: applications.filter(a => a.status === 'Interview').length,
    applied: applications.filter(a => a.status === 'Applied').length,
    rejected: applications.filter(a => a.status === 'Rejected').length
  };

  return (
    <div className="tab-content">
      <h2><BarChart3 size={24} /> Application Dashboard</h2>
      
      <div className="stats-grid">
        <div className="stat-card">
          <h3>{stats.total}</h3>
          <p>Total Applications</p>
        </div>
        <div className="stat-card">
          <h3>{stats.interview}</h3>
          <p>Interviews</p>
        </div>
        <div className="stat-card">
          <h3>{stats.applied}</h3>
          <p>Pending</p>
        </div>
        <div className="stat-card">
          <h3>{stats.rejected}</h3>
          <p>Rejected</p>
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading applications...</div>
      ) : (
        <div className="applications-list">
          <h3><Briefcase size={20} /> Recent Applications</h3>
          <table>
            <thead>
              <tr>
                <th>Company</th>
                <th>Position</th>
                <th>Status</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {applications.map(app => (
                <tr key={app.id}>
                  <td>{app.company}</td>
                  <td>{app.position}</td>
                  <td>
                    <span className="status-badge" style={{backgroundColor: getStatusColor(app.status)}}>
                      {app.status}
                    </span>
                  </td>
                  <td>{app.date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
