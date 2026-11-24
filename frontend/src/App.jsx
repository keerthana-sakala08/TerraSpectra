
import { useEffect, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000/api/analytics/crop-health'

function App() {
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function fetchAnalytics() {
    setLoading(true)
    setError('')

    try {
      const response = await fetch(API_URL)

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      setAnalytics(data)
    } catch (err) {
      setError(err.message || 'Unable to connect to backend.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAnalytics()
  }, [])

  if (loading) {
    return (
      <main className="status-screen">
        <div className="loader"></div>
        <h2>Loading TerraSpectra Analytics...</h2>
      </main>
    )
  }

  if (error) {
    return (
      <main className="status-screen">
        <h2>Unable to load analytics</h2>
        <p>{error}</p>
        <p>Make sure the FastAPI backend is running.</p>
        <button onClick={fetchAnalytics}>Try Again</button>
      </main>
    )
  }

  const risk = analytics.risk_distribution

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">T</div>
          <div>
            <h2>TerraSpectra</h2>
            <span>Smart Agriculture AI</span>
          </div>
        </div>

        <nav className="navigation">
          <p className="nav-heading">MAIN MENU</p>
          <a className="nav-link active" href="#overview">▦ Dashboard</a>
          <a className="nav-link" href="#health">♧ Crop Health</a>
          <a className="nav-link" href="#risk">⚠ Risk Analysis</a>
          <a className="nav-link" href="#about">ⓘ About Dataset</a>
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot"></span>
          Analytics system
          <strong>Connected</strong>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="eyebrow">SMART AGRICULTURE PLATFORM</span>
            <h1>Crop Health Dashboard</h1>
            <p>Monitor crop conditions and analyse stress levels.</p>
          </div>

          <button className="refresh-button" onClick={fetchAnalytics}>
            ↻ Refresh Data
          </button>
        </header>

        <section id="overview" className="welcome-panel">
          <div>
            <span className="panel-tag">DAY 15 ANALYTICS</span>
            <h2>Crop Health Overview</h2>
            <p>
              Your hyperspectral analytics report is connected
              successfully.
            </p>
          </div>
          <div className="welcome-icon">🌱</div>
        </section>

        <section className="stats-grid" id="health">
          <article className="stat-card">
            <div className="stat-top">
              <span>Total Analysed Pixels</span>
              <span className="stat-icon">▦</span>
            </div>
            <h2>{analytics.total_valid_pixels.toLocaleString()}</h2>
            <p>Valid pixels analysed</p>
          </article>

          <article className="stat-card healthy-card">
            <div className="stat-top">
              <span>Healthy Pixels</span>
              <span className="stat-icon">✓</span>
            </div>
            <h2>{analytics.healthy_percentage}%</h2>
            <p>{analytics.healthy_pixels.toLocaleString()} healthy pixels</p>
          </article>

          <article className="stat-card stressed-card">
            <div className="stat-top">
              <span>Stressed Pixels</span>
              <span className="stat-icon">!</span>
            </div>
            <h2>{analytics.stressed_percentage}%</h2>
            <p>{analytics.stressed_pixels.toLocaleString()} stressed pixels</p>
          </article>

          <article className="stat-card risk-card">
            <div className="stat-top">
              <span>High-Risk Pixels</span>
              <span className="stat-icon">⚠</span>
            </div>
            <h2>{risk.high_risk_pixels.toLocaleString()}</h2>
            <p>Probability ≥ 0.70</p>
          </article>
        </section>

        <section className="content-grid">
          <article className="dashboard-card">
            <div className="card-heading">
              <div>
                <h2>Crop Health Distribution</h2>
                <p>Healthy and stressed pixel percentages</p>
              </div>
            </div>

            <div className="distribution">
              <div className="distribution-row">
                <div className="legend-label">
                  <span className="legend-dot healthy-dot"></span>
                  Healthy
                </div>
                <strong>{analytics.healthy_percentage}%</strong>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill healthy-fill"
                  style={{ width: `${analytics.healthy_percentage}%` }}
                ></div>
              </div>

              <div className="distribution-row">
                <div className="legend-label">
                  <span className="legend-dot stressed-dot"></span>
                  Stressed
                </div>
                <strong>{analytics.stressed_percentage}%</strong>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill stressed-fill"
                  style={{ width: `${analytics.stressed_percentage}%` }}
                ></div>
              </div>
            </div>

            <div className="average-box">
              <span>Average Stress Probability</span>
              <strong>{analytics.average_stress_probability}</strong>
            </div>
          </article>

          <article className="dashboard-card" id="risk">
            <div className="card-heading">
              <div>
                <h2>Risk Distribution</h2>
                <p>Pixel counts by probability thresholds</p>
              </div>
            </div>

            <div className="risk-list">
              <div className="risk-item">
                <span className="risk-marker low"></span>
                <div>
                  <strong>Low Risk</strong>
                  <small>Below 0.40</small>
                </div>
                <b>{risk.low_risk_pixels.toLocaleString()}</b>
              </div>

              <div className="risk-item">
                <span className="risk-marker medium"></span>
                <div>
                  <strong>Medium Risk</strong>
                  <small>0.40 to below 0.70</small>
                </div>
                <b>{risk.medium_risk_pixels.toLocaleString()}</b>
              </div>

              <div className="risk-item">
                <span className="risk-marker high"></span>
                <div>
                  <strong>High Risk</strong>
                  <small>0.70 and above</small>
                </div>
                <b>{risk.high_risk_pixels.toLocaleString()}</b>
              </div>
            </div>
          </article>
        </section>

        <section className="dashboard-card dataset-card" id="about">
          <div className="card-heading">
            <div>
              <h2>Dataset Information</h2>
              <p>Details and limitations of this analytics report</p>
            </div>
            <span className="synthetic-badge">Synthetic Dataset</span>
          </div>

          <div className="dataset-details">
            <div>
              <span>Project</span>
              <strong>{analytics.project}</strong>
            </div>
            <div>
              <span>Analytics Type</span>
              <strong>{analytics.analytics_type}</strong>
            </div>
            <div>
              <span>Healthy Class</span>
              <strong>Class 0</strong>
            </div>
            <div>
              <span>Stressed Class</span>
              <strong>Class 1</strong>
            </div>
            <div>
              <span>Real-world Validation</span>
              <strong>{analytics.real_world_validation ? 'Yes' : 'No'}</strong>
            </div>
          </div>

          <div className="limitations">
            <strong>Important limitations</strong>
            <ul>
              {analytics.limitations.map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </div>
        </section>

        <footer className="footer">
          TerraSpectra · AI-Powered Smart Agriculture Platform
          <span>Day 15 Development</span>
        </footer>
      </main>
    </div>
  )
}

export default App
