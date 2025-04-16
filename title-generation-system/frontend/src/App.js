// File: src/App.js
import React, { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/generate-title', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error('Server error');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Error generating title. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>HEADLINER</h1>
        <p>Enter an article or long text to generate a title using argumentation mining</p>
      </header>
      
      <main className="App-main">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="text-input">Article Text:</label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your article or text here..."
              rows={10}
              required
            />
          </div>
          
          <button 
            type="submit" 
            className="submit-button"
            disabled={loading}
          >
            {loading ? 'Generating...' : 'Generate Title'}
          </button>
        </form>
        
        {error && <div className="error-message">{error}</div>}
        
        {result && (
          <div className="result-container">
            <div className="title-result">
              <h2>Generated Title:</h2>
              <div className="title-box">{result.title}</div>
            </div>
            
            <div className="arguments-result">
              <h3>Key Arguments Used:</h3>
              <ul>
                {result.arguments.map((arg, index) => (
                  <li key={index}>{arg}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </main>
      
      <footer className="App-footer">
        <p>Powered by BERT and T5-small</p>
      </footer>
    </div>
  );
}

export default App;