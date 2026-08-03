import { useState, useEffect, useMemo } from 'react'
import { Doughnut, Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS, ArcElement, Tooltip, Legend,
  CategoryScale, LinearScale, BarElement, Title,
} from 'chart.js'
import Sidebar  from '../components/Layout/Sidebar'
import Topbar   from '../components/Layout/Topbar'
import KPICard  from '../components/common/KPICard'
import LoadingSpinner from '../components/common/LoadingSpinner'
import { getAuditHistory } from '../services/auditService'
import { formatINR } from '../utils/formatters'

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title)

export default function AuditHistoryPage() {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('')

  useEffect(() => {
    getAuditHistory()
      .then(r => setData(r.data))
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    if (!data?.logs) return []
    return data.logs.filter(log => {
      const q  = search.toLowerCase()
      const matchQ = !q || [log.diagnosis, log.model_label, log.hospital, log.insurance]
        .join(' ').toLowerCase().includes(q)
      const matchF = !filter
        || (filter === 'Fraud' && log.prediction === 1)
        || (filter === 'Legit' && log.prediction === 0)
      return matchQ && matchF
    })
  }, [data, search, filter])

  const donutData = data ? {
    labels: ['Legitimate', 'Fraud'],
    datasets: [{ data: [data.legit, data.frauds], backgroundColor: ['#10B981', '#EF4444'], borderWidth: 3, borderColor: '#fff', hoverOffset: 8 }],
  } : null

  const barData = data?.model_stats ? {
    labels: data.model_stats.map(m => m.model_label),
    datasets: [{ label: 'Fraud Rate (%)', data: data.model_stats.map(m => m.fraud_rate),
      backgroundColor: data.model_stats.map(m => m.fraud_rate > 50 ? 'rgba(239,68,68,.75)' : 'rgba(37,99,235,.7)'),
      borderRadius: 6, borderSkipped: false }],
  } : null

  const chartOpts = (type) => ({
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { font: { family: 'Inter', weight: '600' }, usePointStyle: type === 'doughnut' } } },
    ...(type === 'bar' ? { scales: {
      y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%', font: { family: 'Inter' } }, grid: { color: '#F1F5F9' } },
      x: { ticks: { font: { family: 'Inter', size: 11 } }, grid: { display: false } },
    }} : {}),
  })

  if (loading) return (
    <div className="app-layout"><Sidebar />
      <main className="main-content"><Topbar title="Audit History" /><LoadingSpinner fullPage label="Loading your audit history…" /></main>
    </div>
  )

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Topbar title="Audit History" />
        <div className="content-wrapper">

          {/* KPIs */}
          <div className="kpi-grid-4">
            <KPICard icon="bi-clipboard2-pulse" label="Total Audits"   value={data?.total ?? 0}         color="blue"   delay={0} />
            <KPICard icon="bi-exclamation-triangle-fill" label="Fraud Detected" value={data?.frauds ?? 0} color="red"    delay={.05} />
            <KPICard icon="bi-shield-fill-check" label="Legitimate"    value={data?.legit ?? 0}          color="green"  delay={.1} />
            <KPICard icon="bi-percent"           label="Fraud Rate"    value={`${data?.fraud_rate ?? 0}%`} color="yellow" delay={.15} />
          </div>

          {/* Charts */}
          {data?.total > 0 && (
            <div className="charts-grid">
              <div className="card fade-in">
                <div className="card-header"><h4>Outcome Distribution</h4><p>Breakdown of your audit verdicts</p></div>
                <div className="card-body"><div className="chart-wrap"><Doughnut data={donutData} options={chartOpts('doughnut')} /></div></div>
              </div>
              <div className="card fade-in fade-in-delay-1">
                <div className="card-header"><h4>Model-wise Fraud Rate</h4><p>Fraud detection rate per intelligence layer</p></div>
                <div className="card-body"><div className="chart-wrap"><Bar data={barData} options={chartOpts('bar')} /></div></div>
              </div>
            </div>
          )}

          {/* Model Breakdown */}
          {data?.model_stats?.length > 0 && (
            <div className="card fade-in" style={{ marginBottom: '2rem' }}>
              <div className="card-header"><h4>Intelligence Layer Breakdown</h4><p>Per-model audit and fraud statistics</p></div>
              <div className="card-body" style={{ padding: '1rem 1.5rem' }}>
                <table className="data-table">
                  <thead><tr><th>Model</th><th>Total</th><th>Fraud</th><th>Legit</th><th style={{ width: 220 }}>Fraud Rate</th></tr></thead>
                  <tbody>
                    {data.model_stats.map((ms, i) => (
                      <tr key={i}>
                        <td><span className="badge-model">{ms.model_label}</span></td>
                        <td><strong>{ms.total}</strong></td>
                        <td style={{ color: 'var(--danger)', fontWeight: 700 }}>{ms.frauds}</td>
                        <td style={{ color: 'var(--success)', fontWeight: 700 }}>{ms.legit}</td>
                        <td>
                          <div className="model-stat-row">
                            <div className="bar-track"><div className="bar-fill" style={{ width: `${ms.fraud_rate}%` }} /></div>
                            <span style={{ fontWeight: 700, color: 'var(--danger)', minWidth: 42 }}>{ms.fraud_rate}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Log Table */}
          <div className="card fade-in">
            <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div><h4><i className="bi bi-list-check" style={{ marginRight: 8 }} />Full Audit Log</h4><p>All claims you have submitted</p></div>
              <span style={{ fontSize: '.85rem', color: 'var(--text-muted)' }}>{data?.total ?? 0} records</span>
            </div>
            <div className="card-body">
              {data?.logs?.length > 0 ? (
                <>
                  <div className="controls-bar">
                    <div className="search-wrap">
                      <i className="bi bi-search" />
                      <input placeholder="Search by diagnosis, model, hospital…" value={search} onChange={e => setSearch(e.target.value)} />
                    </div>
                    <select className="filter-select" value={filter} onChange={e => setFilter(e.target.value)}>
                      <option value="">All Verdicts</option>
                      <option value="Fraud">Fraud</option>
                      <option value="Legit">Legitimate</option>
                    </select>
                  </div>

                  <table className="data-table">
                    <thead>
                      <tr><th>#ID</th><th>Model</th><th>Verdict</th><th>Claim (₹)</th><th>Diagnosis</th><th>Hospital</th><th>Insurance</th><th>Days</th><th>Timestamp</th></tr>
                    </thead>
                    <tbody>
                      {filtered.length === 0 ? (
                        <tr><td colSpan={9} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>No matching records</td></tr>
                      ) : filtered.map(log => (
                        <tr key={log.id}>
                          <td className="id-cell">#{log.id}</td>
                          <td><span className="badge-model">{log.model_label}</span></td>
                          <td>
                            {log.prediction === 1
                              ? <span className="badge badge-fraud"><i className="bi bi-exclamation-circle-fill" />Fraud</span>
                              : <span className="badge badge-legit"><i className="bi bi-check-circle-fill" />Legitimate</span>
                            }
                          </td>
                          <td style={{ fontWeight: 700 }}>{formatINR(log.claim_amount)}</td>
                          <td><code style={{ fontSize: '.84rem', background: '#F1F5F9', padding: '.15rem .4rem', borderRadius: 4 }}>{log.diagnosis}</code></td>
                          <td>{log.hospital}</td>
                          <td>{log.insurance}</td>
                          <td>{log.days}</td>
                          <td style={{ color: 'var(--text-muted)', fontSize: '.82rem', whiteSpace: 'nowrap' }}>{log.created_at || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : (
                <div className="empty-state">
                  <i className="bi bi-clipboard2-x" />
                  <h5>No audits yet</h5>
                  <p>Submit your first claim from the <a href="#/dashboard" style={{ color: 'var(--primary)', fontWeight: 600 }}>Audit Dashboard</a>.</p>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}
