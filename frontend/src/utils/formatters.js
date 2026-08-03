/** Format a number as Indian Rupees: ₹1,23,456 */
export function formatINR(value) {
  const num = parseFloat(value)
  if (isNaN(num)) return '—'
  return '₹' + num.toLocaleString('en-IN', { maximumFractionDigits: 0 })
}

/** Return a short label for a model key */
export function modelLabel(key, display) {
  return display?.[key] || key
}

/** Truncate long text */
export function truncate(str, max = 60) {
  if (!str) return '—'
  return str.length > max ? str.slice(0, max) + '…' : str
}
