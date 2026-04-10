import React, { useState } from 'react';
import { Mic, MicOff, Play, RotateCcw } from 'lucide-react';
import { mockApi } from '../services/mockApi';

export default function MockInterview() {
  const [isRecording, setIsRecording] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);

  const startInterview = async () => {
    setLoading(true);
    const result = await mockApi.conductMockInterview();
    setCurrentQuestion(result.question);
    setFeedback(null);
    setLoading(false);
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    if (isRecording) {
      setFeedback("Good answer! Consider adding more specific examples and metrics to strengthen your response.");
    }
  };

  const nextQuestion = async () => {
    setLoading(true);
    const result = await mockApi.conductMockInterview();
    setCurrentQuestion(result.question);
    setFeedback(null);
    setIsRecording(false);
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <h2>Mock Interview Practice</h2>
      
      {!currentQuestion ? (
        <div className="interview-start">
          <p>Practice your interview skills with AI-powered mock interviews</p>
          <button onClick={startInterview} disabled={loading} className="start-btn">
            <Play size={20} /> {loading ? 'Loading...' : 'Start Interview'}
          </button>
        </div>
      ) : (
        <div className="interview-active">
          <div className="question-card">
            <h3>Question:</h3>
            <p>{currentQuestion}</p>
          </div>

          <div className="recording-controls">
            <button
              onClick={toggleRecording}
              className={`record-btn ${isRecording ? 'recording' : ''}`}
            >
              {isRecording ? <MicOff size={32} /> : <Mic size={32} />}
            </button>
            <p>{isRecording ? 'Recording... Click to stop' : 'Click to start recording your answer'}</p>
          </div>

          {feedback && (
            <div className="feedback-card">
              <h4>Feedback:</h4>
              <p>{feedback}</p>
              <button onClick={nextQuestion} className="next-btn">
                <RotateCcw size={18} /> Next Question
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
