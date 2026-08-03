import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { useAuth } from '../context/AuthContext'
import { loginUser } from '../services/authService'

export default function LoginPage() {
  const [form, setForm]       = useState({ username: '', password: '' })
  const [errors, setErrors]   = useState({})
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate   = useNavigate()

  const validate = () => {
    const e = {}
    if (!form.username.trim()) e.username = 'Username is required'
    if (!form.password.trim()) e.password = 'Password is required'
    setErrors(e)
    return Object.keys(e).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!validate()) return
    setLoading(true)
    try {
      const res = await loginUser(form)
      login(res.data)
      toast.success(`Welcome back, ${res.data.user}!`)
      navigate(res.data.role === 'admin' ? '/admin' : '/dashboard')
    } catch (err) {
      const msg = err.response?.data?.error || 'Login failed. Please try again.'
      toast.error(msg)
      setErrors({ server: msg })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      {/* ── Left Brand Panel ── */}
      <div className="auth-brand">
        <div className="auth-brand-logo">
          <i className="bi bi-shield-check" />
          <h2>Medicare<br />Audit Intelligence</h2>
        </div>
        <h1>Detect Fraud.<br />Protect Healthcare.</h1>
        <p>
          AI-powered claim analysis using 8 intelligence architectures —
          from Random Forest to Hybrid Deep Learning — calibrated for
          India's healthcare ecosystem.
        </p>
        <div className="auth-feature">
          <i className="bi bi-cpu-fill" />
          8 ML Models including CNN, Transformer &amp; GNN
        </div>
        <div className="auth-feature">
          <i className="bi bi-shield-fill-check" />
          Expert rule engine for Indian insurance schemes
        </div>
        <div className="auth-feature">
          <i className="bi bi-graph-up-arrow" />
          SHAP explainability for every audit decision
        </div>
        <div className="auth-feature">
          <i className="bi bi-currency-rupee" />
          INR-calibrated fraud thresholds (PM-JAY, ESI, CGHS)
        </div>
      </div>

      {/* ── Right Form Panel ── */}
      <div className="auth-form-side">
        <div className="auth-card fade-in">
          <h2>Welcome back</h2>
          <p className="auth-subtitle">Sign in to your audit dashboard</p>

          {errors.server && (
            <div className="alert-box error">
              <i className="bi bi-exclamation-circle-fill" /> {errors.server}
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate>
            <div className="form-group">
              <label className="form-label" htmlFor="username">Username</label>
              <input
                id="username"
                className={`form-control${errors.username ? ' error' : ''}`}
                type="text"
                placeholder="Enter your username"
                value={form.username}
                onChange={e => setForm({ ...form, username: e.target.value })}
                autoFocus
              />
              {errors.username && <div className="form-error"><i className="bi bi-exclamation-circle" />{errors.username}</div>}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="password">Password</label>
              <input
                id="password"
                className={`form-control${errors.password ? ' error' : ''}`}
                type="password"
                placeholder="Enter your password"
                value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
              />
              {errors.password && <div className="form-error"><i className="bi bi-exclamation-circle" />{errors.password}</div>}
            </div>

            <button
              type="submit"
              className="btn btn-primary btn-full btn-lg"
              disabled={loading}
              style={{ marginTop: '.5rem' }}
            >
              {loading ? <><span className="spinner spinner-sm" /> Authenticating…</> : <><i className="bi bi-box-arrow-in-right" /> Sign In</>}
            </button>
          </form>

          <p className="auth-link">
            Don't have an account? <Link to="/register">Create one</Link>
          </p>

          <div style={{
            marginTop: '2rem', padding: '1rem', background: '#F8FAFC',
            borderRadius: '8px', fontSize: '.82rem', color: 'var(--text-muted)'
          }}>
            <strong style={{ color: 'var(--text)' }}>Admin access:</strong> username <code>admin</code> / password <code>admin123</code>
          </div>
        </div>
      </div>
    </div>
  )
}
