import api from './api'

export const getModels      = ()     => api.get('/api/models')
export const runPredict     = (data) => api.post('/api/predict', data)
export const getAuditHistory = ()    => api.get('/api/audit-history')
export const getAdminData   = ()     => api.get('/api/admin')
