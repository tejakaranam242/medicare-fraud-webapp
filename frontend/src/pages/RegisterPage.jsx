import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { registerUser } from '../services/authService'

export default function RegisterPage() {
  const [form, setForm]       = useState({ username: '', password: '', confirm: '' })
  const [errors, setErrors]   = useState({})
  const [loading, setLoading] = useState(false)
  const navigate               = useNavigate()

  const validate = () => {
    const e = {}
    if (!form.username.trim())          e.username = 'Username is required'
    else if (form.username.length < 3)  e.username = 'Username must be at least 3 characters'
    if (!form.password.trim())          e.password = 'Password is required'
    else if (form.password.length < 4)  e.password = 'Password must be at least 4 characters'
    if (form.password !== form.confirm) e.confirm  = 'Passwords do not match'
    setErrors(e)
    return Object.keys(e).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!validate()) return
    setLoading(true)
    try {
      await registerUser({ username: form.username, password: form.password })
      toast.success('Account created! Please sign in.')
      navigate('/login')
    } catch (err) {
      const msg = err.response?.data?.error || 'Registration failed. Please try again.'
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
        <h1>Join the Audit<br />Intelligence Platform</h1>
        <p>
          Create your account to start running AI-powered fraud detection
          on Indian Medicare claims. Your analysis history and reports are
          securely stored.
        </p>
        <div className="auth-feature">
          <i className="bi bi-person-check-fill" />
          Private audit history — scoped to your account
        </div>
        <div className="auth-feature">
          <i className="bi bi-lock-fill" />
          Secure session-based authentication
        </div>
        <div className="auth-feature">
          <i className="bi bi-bar-chart-fill" />
          Personal analytics and fraud rate dashboard
        </div>
      </div>

      {/* ── Right Form Panel ── */}
      <div className="auth-form-side">
        <div className="auth-card fade-in">
          <h2>Create account</h2>
          <p className="auth-subtitle">Start auditing claims in seconds</p>

          {errors.server && (
            <div className="alert-box error">
              <i className="bi bi-exclamation-circle-fill" /> {errors.server}
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate>
            <div className="form-group">
              <label className="form-label" htmlFor="reg-username">Username</label>
              <input
                id="reg-username"
                className={`form-control${errors.username ? ' error' : ''}`}
                type="text"
                placeholder="Choose a username (min 3 chars)"
                value={form.username}
                onChange={e => setForm({ ...form, username: e.target.value })}
                autoFocus
              />
              {errors.username && <div className="form-error"><i className="bi bi-exclamation-circle" />{errors.username}</div>}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="reg-password">Password</label>
              <input
                id="reg-password"
                className={`form-control${errors.password ? ' error' : ''}`}
                type="password"
                placeholder="At least 4 characters"
                value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
              />
              {errors.password && <div className="form-error"><i className="bi bi-exclamation-circle" />{errors.password}</div>}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="reg-confirm">Confirm Password</label>
              <input
                id="reg-confirm"
                className={`form-control${errors.confirm ? ' error' : ''}`}
                type="password"
                placeholder="Repeat your password"
                value={form.confirm}
                onChange={e => setForm({ ...form, confirm: e.target.value })}
              />
              {errors.confirm && <div className="form-error"><i className="bi bi-exclamation-circle" />{errors.confirm}</div>}
            </div>

            <button type="submit" className="btn btn-primary btn-full btn-lg" disabled={loading} style={{ marginTop: '.5rem' }}>
              {loading ? <><span className="spinner spinner-sm" /> Creating account…</> : <><i className="bi bi-person-plus-fill" /> Create Account</>}
            </button>
          </form>

          <p className="auth-link">
            Already have an account? <Link to="/login">Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  )
}
