import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import toast from 'react-hot-toast'

export default function Sidebar() {
  const { user, role, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    await logout()
    toast.success('Signed out successfully')
    navigate('/login')
  }

  const navClass = ({ isActive }) => `nav-link${isActive ? ' active' : ''}`

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <i className="bi bi-shield-check" />
        <h5>Medicare<br />Audit Intel</h5>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/dashboard" className={navClass}>
          <i className="bi bi-file-earmark-medical" /> New Claim Audit
        </NavLink>

        {role === 'admin' && (
          <NavLink to="/admin" className={navClass}>
            <i className="bi bi-graph-up-arrow" /> Command Center
          </NavLink>
        )}

        {role === 'user' && (
          <>
            <NavLink to="/history" className={navClass}>
              <i className="bi bi-clock-history" /> Audit History
            </NavLink>
          </>
        )}

        <NavLink to="/result" className={navClass}>
          <i className="bi bi-clipboard-data" /> Last Report
        </NavLink>
      </nav>

      <div className="sidebar-footer">
        <button onClick={handleLogout} className="btn-logout">
          <i className="bi bi-box-arrow-left" /> Sign Out
        </button>
      </div>
    </aside>
  )
}
