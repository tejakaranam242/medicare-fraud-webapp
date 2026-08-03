import { useAuth } from '../../context/AuthContext'
import { Navigate } from 'react-router-dom'
import LoadingSpinner from './LoadingSpinner'

/** Wraps a route — redirects to /login if unauthenticated, /dashboard if non-admin hits admin-only route */
export default function ProtectedRoute({ children, adminOnly = false }) {
  const { user, role, loading } = useAuth()

  if (loading) return <LoadingSpinner fullPage />

  if (!user) return <Navigate to="/login" replace />

  if (adminOnly && role !== 'admin') return <Navigate to="/dashboard" replace />

  return children
}
