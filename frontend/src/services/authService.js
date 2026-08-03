import api from './api'

export const loginUser     = (data) => api.post('/api/login', data)
export const registerUser  = (data) => api.post('/api/register', data)
export const logoutUser    = ()     => api.post('/api/logout')
export const getSession    = ()     => api.get('/api/session')
