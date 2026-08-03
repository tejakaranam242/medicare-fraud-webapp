import { useAuth } from '../../context/AuthContext'

export default function Topbar({ title }) {
  const { user, role } = useAuth()
  const initials = user ? user[0].toUpperCase() : '?'

  return (
    <header className="topbar">
      <h2 className="topbar-title">{title}</h2>
      <div className="topbar-right">
        <span className="topbar-user">
          {role === 'admin' ? 'Super Admin' : `Auditor: ${user}`}
        </span>
        <div className={`avatar ${role === 'admin' ? 'admin' : ''}`}>
          {role === 'admin' ? 'SA' : initials}
        </div>
      </div>
    </header>
  )
}
