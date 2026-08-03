export default function LoadingSpinner({ fullPage = false, label = 'Loading…' }) {
  if (fullPage) {
    return (
      <div className="page-loading">
        <div className="spinner" />
        <span style={{ color: 'var(--text-muted)', fontSize: '.9rem' }}>{label}</span>
      </div>
    )
  }
  return <div className="spinner spinner-sm" />
}
