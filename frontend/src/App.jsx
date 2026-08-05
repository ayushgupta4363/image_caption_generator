  import React, { useState } from 'react';
  import './App.css';

  // Live GCP Cloud Run Endpoint
  const API_URL = 'https://image-caption-api-694787194678.us-central1.run.app/predict';

  function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [imageMeta, setImageMeta] = useState(null);
    const [caption, setCaption] = useState('');
    const [loading, setLoading] = useState(false);
    const [copied, setCopied] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [error, setError] = useState(null);

    // 1. Handle File Upload & Extract Metadata
    const handleFileChange = (file) => {
      if (!file || !file.type.startsWith('image/')) {
        setError('Please select a valid image file (JPG, PNG, WEBP).');
        return;
      }

      setError(null);
      setCaption('');
      setSelectedFile(file);

      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);

      // Extract image dimensions
      const img = new Image();
      img.src = objectUrl;
      img.onload = () => {
        setImageMeta({
          name: file.name,
          size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
          dimensions: `${img.width} × ${img.height} px`,
          type: file.type.split('/')[1].toUpperCase()
        });
      };
    };

    const handleInputChange = (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileChange(e.target.files[0]);
      }
    };

    const handleDragOver = (e) => e.preventDefault();
    const handleDrop = (e) => {
      e.preventDefault();
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileChange(e.dataTransfer.files[0]);
      }
    };

    // 2. Fetch Prediction from Google Cloud Run
    const generateCaption = async () => {
      if (!selectedFile) return;

      setLoading(true);
      setCaption('');
      setError(null);

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch(API_URL, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server returned status code ${response.status}`);
        }

        const data = await response.json();

        // Flexible extraction to match various FastAPI JSON response keys
        const predictedCaption = data.caption || data.result || data.generated_caption;

        if (predictedCaption) {
          setCaption(predictedCaption);
        } else if (data.error) {
          setError(data.error);
        } else {
          setError('Received an unexpected response format from server.');
        }
      } catch (err) {
        console.error(err);
        setError('Failed to connect to Google Cloud Run backend. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    // 3. Audio Readout (Text-to-Speech)
    const speakCaption = () => {
      if (!caption || !('speechSynthesis' in window)) return;

      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(caption);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);

      window.speechSynthesis.speak(utterance);
    };

    // 4. Copy Caption
    const copyCaption = () => {
      if (!caption) return;
      navigator.clipboard.writeText(caption);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    };

    return (
      <div className="layout-wrapper">
        {/* Navbar */}
        <nav className="navbar">
          <div className="logo-badge">
            <span className="logo-icon">✨</span>
            <span className="logo-text">VisionCaption AI</span>
          </div>
        </nav>

        <div className="app-container">
          {/* Header */}
          <header className="hero-section">
            <h1>
              Transform Images into <span className="gradient-text">Captivating Stories</span>
            </h1>
          </header>

          {/* Workspace Grid */}
          <div className="workspace-grid">
            {/* Upload & Preview Card */}
            <div className="card upload-card">
              <div 
                className={`drop-zone ${previewUrl ? 'has-preview' : ''}`}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                {previewUrl ? (
                  <div className="preview-container">
                    <img src={previewUrl} alt="Preview" className="preview-image" />
                    <button 
                      className="reset-btn"
                      onClick={() => { setSelectedFile(null); setPreviewUrl(null); setCaption(''); setImageMeta(null); }}
                    >
                      ✕ Clear Image
                    </button>
                  </div>
                ) : (
                  <div className="upload-prompt">
                    <div className="icon-circle">
                      <svg className="upload-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <h3>Drag & Drop your image</h3>
                    <p>Supports JPG, PNG, WEBP</p>
                    <label className="file-browse-btn">
                      Browse Computer
                      <input type="file" accept="image/*" onChange={handleInputChange} className="hidden-input" />
                    </label>
                  </div>
                )}
              </div>

              {/* Image Metadata Badge */}
              {imageMeta && (
                <div className="meta-badge-grid">
                  <div className="meta-item"><span>File:</span> {imageMeta.name}</div>
                  <div className="meta-item"><span>Size:</span> {imageMeta.size}</div>
                  <div className="meta-item"><span>Res:</span> {imageMeta.dimensions}</div>
                </div>
              )}

              {/* Action Button */}
              <button 
                className={`generate-action-btn ${loading ? 'loading' : ''}`}
                onClick={generateCaption} 
                disabled={!selectedFile || loading}
              >
                {loading ? (
                  <span className="loader-box">
                    <span className="spinner"></span> Running Attention Model...
                  </span>
                ) : (
                  '⚡ Generate AI Caption'
                )}
              </button>

              {error && <div className="error-toast">{error}</div>}
            </div>

            {/* Result & Tools Card */}
            <div className="card result-card">
              <h2>Model Output</h2>
              {caption ? (
                <div className="caption-output-box">
                  <blockquote className="caption-quote">
                    "{caption}"
                  </blockquote>

                  <div className="action-button-group">
                    <button className={`btn-secondary ${isSpeaking ? 'active-speech' : ''}`} onClick={speakCaption}>
                      {isSpeaking ? '🔊 Speaking...' : '🔊 Listen Audio'}
                    </button>
                    <button className="btn-secondary" onClick={copyCaption}>
                      {copied ? '✅ Copied!' : '📋 Copy Text'}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="empty-result-state">
                  <span className="sparkle-icon">💡</span>
                  <p>Upload an image and click <strong>Generate AI Caption</strong> to see the predicted sentence here.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  export default App;