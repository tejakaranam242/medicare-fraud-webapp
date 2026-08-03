import { useLocation, useNavigate } from 'react-router-dom'
import Sidebar from '../components/Layout/Sidebar'
import Topbar  from '../components/Layout/Topbar'
import { formatINR } from '../utils/formatters'

export default function ResultPage() {
  const location = useLocation()
  const navigate  = useNavigate()

  // Prefer navigation state, fallback to sessionStorage
  const result = location.state?.result
    || (() => { try { return JSON.parse(sessionStorage.getItem('auditResult')) } catch { return null } })()

  if (!result) {
    return (
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Topbar title="Audit Report" />
          <div className="content-wrapper">
            <div className="empty-state">
              <i className="bi bi-clipboard2-x" />
              <h5>No result found</h5>
              <p>Submit a claim from the <a href="#/dashboard" style={{ color: 'var(--primary)', fontWeight: 600 }}>audit dashboard</a> first.</p>
            </div>
          </div>
        </main>
      </div>
    )
  }

  const isFraud = result.prediction === 1

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Topbar title="Audit Summary Report" />
        <div className="content-wrapper">

          {/* Back */}
          <button className="btn btn-ghost btn-sm" style={{ marginBottom: '1.25rem' }} onClick={() => navigate('/dashboard')}>
            <i className="bi bi-arrow-left" /> Back to Dashboard
          </button>

          {/* Verdict Banner */}
          <div className={`verdict-banner ${isFraud ? 'fraud' : 'legit'} fade-in`}>
            <div className="verdict-left">
              <span className="verdict-icon">
                <i className={`bi ${isFraud ? 'bi-exclamation-octagon-fill' : 'bi-check-circle-fill'}`} />
              </span>
              <div>
                <div className="verdict-title">{result.prediction_label}</div>
                <div className="verdict-sub">Model: <strong>{result.current_model}</strong></div>
              </div>
            </div>
            <button className="btn" style={{ background: 'rgba(255,255,255,.2)', color: '#fff', border: '1px solid rgba(255,255,255,.4)' }}
              onClick={() => window.print()}>
              <i className="bi bi-printer" /> Print Report
            </button>
          </div>

          {/* Auditor intelligence note */}
          {result.findings?.length > 0 && (
            <div className="alert-box warn fade-in fade-in-delay-1" style={{ marginBottom: '1.5rem' }}>
              <i className="bi bi-lightbulb-fill" />
              <span>
                The expert heuristic layer flagged this claim due to <strong>profile inconsistency</strong>.
                Manual verification of the original patient chart is highly recommended.
              </span>
            </div>
          )}

          {/* Risk Findings */}
          <div className="card fade-in fade-in-delay-1" style={{ marginBottom: '1.5rem', borderLeft: '5px solid var(--danger)' }}>
            <div className="card-header">
              <h4 style={{ color: 'var(--danger)', display: 'flex', gap: 8, alignItems: 'center' }}>
                <i className="bi bi-search" /> Risk Analysis Findings
              </h4>
            </div>
            <div className="card-body">
              {result.findings?.length > 0 ? (
                result.findings.map((f, i) => (
                  <div key={i} className="finding-item">
                    <span className="finding-dot"><i className="bi bi-dot" /></span>
                    <span>{f}</span>
                  </div>
                ))
              ) : (
                <p style={{ color: 'var(--text-muted)' }}>No specific anomalies detected for this claim profile.</p>
              )}
            </div>
          </div>

          {/* Confidence + Meta */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1.25rem', marginBottom: '1.5rem' }}>
            {/* Confidence */}
            <div className="card fade-in fade-in-delay-2">
              <div className="card-body">
                <div className="section-title">Intelligence Confidence</div>
                <div className="confidence-ring">
                  <div className="confidence-number">
                    {result.confidence}<span>%</span>
                  </div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '.85rem' }}>Probability Assessment</p>
                  <div className="confidence-bar">
                    <div
                      className={`confidence-fill ${isFraud ? 'fraud' : 'legit'}`}
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Claim metadata */}
            <div className="card fade-in fade-in-delay-2">
              <div className="card-body">
                <div className="section-title">Extracted Features Snapshot</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 2rem' }}>
                  <ul className="meta-list">
                    <li><span className="meta-label">Claim Amount</span><span className="meta-value">{formatINR(result.input_data?.Claim_Amount)}</span></li>
                    <li><span className="meta-label">Days Admitted</span><span className="meta-value">{result.input_data?.Days_Admitted || '—'}</span></li>
                    <li><span className="meta-label">Procedures</span><span className="meta-value">{result.input_data?.Number_of_Procedures || '—'}</span></li>
                  </ul>
                  <ul className="meta-list">
                    <li><span className="meta-label">Diagnosis</span><span className="meta-value">{result.input_data?.Diagnosis_Code || '—'}</span></li>
                    <li><span className="meta-label">Hospital Type</span><span className="meta-value">{result.input_data?.Hospital_Type || '—'}</span></li>
                    <li><span className="meta-label">Insurance</span><span className="meta-value">{result.input_data?.Insurance_Type || '—'}</span></li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* AI Narrative */}
          {result.narrative && (
            <div className="card fade-in fade-in-delay-3" style={{ marginBottom: '1.5rem', borderLeft: '5px solid var(--primary)' }}>
              <div className="card-body">
                <div className="section-title" style={{ color: 'var(--primary)' }}>
                  <i className="bi bi-robot" style={{ marginRight: 6 }} />AI Auditor Narrative
                </div>
                <p style={{ color: '#334155', lineHeight: 1.7, margin: 0, fontStyle: 'italic' }}>"{result.narrative}"</p>
              </div>
            </div>
          )}

          {/* SHAP Explainability */}
          <div className="card fade-in fade-in-delay-3">
            <div className="card-header">
              <h4>Explainability Matrix (SHAP)</h4>
              <p>Red factors push toward fraud; blue factors indicate legitimate behaviour patterns.</p>
            </div>
            <div className="card-body">
              {result.shap_plot ? (
                <div className="shap-box">
                  <img src={`/${result.shap_plot}`} alt="SHAP Explainability Waterfall" />
                  <div style={{ marginTop: '1rem' }}>
                    <a href={`/${result.shap_plot}`} download className="btn btn-outline btn-sm">
                      <i className="bi bi-download" /> Export Matrix
                    </a>
                  </div>
                </div>
              ) : (
                <div className="alert-box info">
                  <i className="bi bi-info-circle-fill" />
                  Explainability mapping is not available for this architecture path.
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}
