import axios from 'axios'

/** Axios instance — cookies are automatically sent for same-origin requests */
const api = axios.create({
  baseURL: '',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
})

export default api
