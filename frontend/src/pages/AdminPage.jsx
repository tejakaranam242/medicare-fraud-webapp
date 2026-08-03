import { useState, useEffect, useMemo } from 'react'
import { Doughnut } from 'react-chartjs-2'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'
import Sidebar  from '../components/Layout/Sidebar'
import Topbar   from '../components/Layout/Topbar'
import KPICard  from '../components/common/KPICard'
import LoadingSpinner from '../components/common/LoadingSpinner'
import { getAdminData } from '../services/auditService'
import { formatINR } from '../utils/formatters'

ChartJS.register(ArcElement, Tooltip, Legend)

export default function AdminPage() {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  useEffect(() => {
    getAdminData()
      .then(r => setData(r.data))
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [])

  const filteredLogs = useMemo(() => {
    if (!data?.logs) return []
    const q = search.toLowerCase()
    return !q ? data.logs : data.logs.filter(l =>
      [l.username, l.model_label, String(l.id)].join(' ').toLowerCase().includes(q)
    )
  }, [data, search])

  const donutData = data ? {
    labels: ['Legitimate', 'Fraud'],
    datasets: [{ data: [data.legit, data.frauds], backgroundColor: ['#10B981', '#EF4444'], borderWidth: 3, borderColor: '#fff', hoverOffset: 8 }],
  } : null

  if (loading) return (
    <div className="app-layout"><Sidebar />
      <main className="main-content"><Topbar title="Command Center" /><LoadingSpinner fullPage label="Loading admin data…" /></main>
    </div>
  )

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Topbar title="Administrative Command Center" />
        <div className="content-wrapper">

          {/* KPIs */}
          <div className="kpi-grid-3">
            <KPICard icon="bi-activity"            label="Total Audits Processed" value={data?.total ?? 0}       color="blue"  delay={0} />
            <KPICard icon="bi-exclamation-triangle" label="Detected Anomalies"     value={data?.frauds ?? 0}      color="red"   delay={.05} />
            <KPICard icon="bi-pie-chart"            label="Global Fraud Ratio"     value={`${data?.fraud_rate ?? 0}%`} color="green" delay={.1} />
          </div>

          {/* Donut + Export buttons row */}
          <div className="charts-grid" style={{ gridTemplateColumns: '1fr 2fr' }}>
            <div className="card fade-in">
              <div className="card-header"><h4>Global Verdict Split</h4></div>
              <div className="card-body">
                {data?.total > 0
                  ? <div className="chart-wrap"><Doughnut data={donutData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { font: { family: 'Inter', weight: '600' }, usePointStyle: true } } } }} /></div>
                  : <div className="empty-state" style={{ padding: '2rem' }}><i className="bi bi-pie-chart-fill" /><p>No data yet</p></div>
                }
              </div>
            </div>
            <div className="card fade-in fade-in-delay-1">
              <div className="card-header"><h4>Export &amp; Actions</h4><p>Download system data as CSV</p></div>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <a href="/download_predictions" className="btn btn-outline" style={{ textDecoration: 'none' }}>
                  <i className="bi bi-cloud-download" /> Export Prediction Logs (CSV)
                </a>
                <a href="/download_users" className="btn btn-outline" style={{ textDecoration: 'none' }}>
                  <i className="bi bi-people" /> Export User Directory (CSV)
                </a>
                <div className="alert-box info" style={{ marginBottom: 0 }}>
                  <i className="bi bi-info-circle-fill" />
                  CSV exports use the existing Flask download endpoints.
                </div>
              </div>
            </div>
          </div>

          {/* Prediction Logs */}
          <div className="card fade-in" style={{ marginBottom: '2rem' }}>
            <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div><h4>Recent Intelligence Logs</h4><p>Last 50 predictions across all users</p></div>
              <a href="/download_predictions" className="btn btn-outline btn-sm" style={{ textDecoration: 'none' }}>
                <i className="bi bi-cloud-download" /> Export CSV
              </a>
            </div>
            <div style={{ padding: '1rem 1.5rem 0' }}>
              <div className="controls-bar">
                <div className="search-wrap">
                  <i className="bi bi-search" />
                  <input placeholder="Search by auditor, model, ID…" value={search} onChange={e => setSearch(e.target.value)} />
                </div>
              </div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr><th>Audit ID</th><th>Initiating Auditor</th><th>Model Engine</th><th>Final Classification</th></tr>
                </thead>
                <tbody>
                  {filteredLogs.length === 0 ? (
                    <tr><td colSpan={4} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>No records found</td></tr>
                  ) : filteredLogs.map(log => (
                    <tr key={log.id}>
                      <td className="id-cell">#AUDIT-{log.id}</td>
                      <td><i className="bi bi-person-circle" style={{ marginRight: 8, color: 'var(--text-muted)' }} />{log.username}</td>
                      <td><span className="badge-model">{log.model_label}</span></td>
                      <td>
                        {log.prediction === 1
                          ? <span className="badge badge-fraud"><i className="bi bi-x-circle-fill" />Flagged Fraud</span>
                          : <span className="badge badge-legit"><i className="bi bi-check-circle-fill" />Legitimate</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* User Directory */}
          <div className="card fade-in">
            <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div><h4>System Access Directory</h4><p>All registered user accounts</p></div>
              <a href="/download_users" className="btn btn-outline btn-sm" style={{ textDecoration: 'none' }}>
                <i className="bi bi-cloud-download" /> Export CSV
              </a>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead><tr><th>User ID</th><th>Account Alias</th><th>Registration Timestamp</th></tr></thead>
                <tbody>
                  {!data?.users?.length ? (
                    <tr><td colSpan={3} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>No users found</td></tr>
                  ) : data.users.map(u => (
                    <tr key={u.id}>
                      <td className="id-cell">UID-{u.id}</td>
                      <td style={{ fontWeight: 500 }}>{u.username}</td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '.88rem' }}>{u.created_at || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}
