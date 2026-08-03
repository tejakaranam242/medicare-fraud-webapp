/** A single KPI stat card */
export default function KPICard({ icon, label, value, color = 'blue', delay = 0 }) {
  return (
    <div className={`kpi-card fade-in`} style={{ animationDelay: `${delay}s` }}>
      <div className={`kpi-icon ${color}`}>
        <i className={`bi ${icon}`} />
      </div>
      <div>
        <div className="kpi-label">{label}</div>
        <div className="kpi-value" style={{ color: colorMap[color] }}>{value}</div>
      </div>
    </div>
  )
}

const colorMap = {
  blue:   'var(--primary)',
  red:    'var(--danger)',
  green:  'var(--success)',
  yellow: 'var(--warning)',
}
