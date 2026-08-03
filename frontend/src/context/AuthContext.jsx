import { createContext, useContext, useState, useEffect } from 'react'
import api from '../services/api'

const AuthContext = createContext(null)

/** Provides user session state to the entire app */
export function AuthProvider({ children }) {
  const [user, setUser]     = useState(null)
  const [role, setRole]     = useState(null)
  const [loading, setLoading] = useState(true)

  // Restore session from Flask cookie on mount
  useEffect(() => {
    api.get('/api/session')
      .then(res => { setUser(res.data.user); setRole(res.data.role) })
      .catch(() => { setUser(null); setRole(null) })
      .finally(() => setLoading(false))
  }, [])

  const login = (userData) => {
    setUser(userData.user)
    setRole(userData.role)
  }

  const logout = async () => {
    try { await api.post('/api/logout') } catch (_) {}
    setUser(null)
    setRole(null)
  }

  return (
    <AuthContext.Provider value={{ user, role, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
