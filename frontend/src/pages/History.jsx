import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { deleteInspection, listInspections } from '../api'
import { Search, ArrowUpDown, Trash2, ExternalLink, History as HistoryIcon } from 'lucide-react'

export default function History() {
  const navigate = useNavigate()
  const [all, setAll] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [sortDescending, setSortDescending] = useState(true)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const data = await listInspections({ limit: 100 })
      setAll(data.items || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const rows = useMemo(() => {
    return (all || [])
      .filter((item) => {
        if (typeFilter !== 'all' && item.inspection_type !== typeFilter) return false
        if (!query.trim()) return true
        const haystack = [item.id, item.inspection_type, item.status, item.predicted_class].join(' ').toLowerCase()
        return haystack.includes(query.toLowerCase())
      })
      .sort((a, b) => {
        const left = new Date(a.created_at).getTime()
        const right = new Date(b.created_at).getTime()
        return sortDescending ? right - left : left - right
      })
  }, [all, typeFilter, query, sortDescending])

  async function remove(id) {
    if (!window.confirm('Delete this inspection record from the database?')) return
    try {
      await deleteInspection(id)
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  if (loading) {
    return (
      <div className="page-shell">
        <div className="spinner-center"><div className="spinner" /></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="page-shell">
        <div className="panel panel-pad">
          <p style={{ color: 'var(--danger)' }}>{error}</p>
          <button className="button button-secondary" style={{ marginTop: '0.75rem' }} onClick={load}>Retry</button>
        </div>
      </div>
    )
  }

  return (
    <div className="page-shell">
      <div className="page-header">
        <h1>Inspection History</h1>
        <p>Browse, search, and manage all stored inspection records.</p>
      </div>

      {/* ── Filters ───────────────────────────────────────────── */}
      <div className="surface-band" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-end' }}>
          <div style={{ flex: '1 1 220px' }}>
            <div className="stat-label">Search</div>
            <div style={{ position: 'relative' }}>
              <Search size={15} style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-soft)' }} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="ID, type, status, class..."
                style={{ paddingLeft: '2.2rem' }}
                id="history-search"
              />
            </div>
          </div>
          <div style={{ flex: '0 0 160px' }}>
            <div className="stat-label">Type</div>
            <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} id="history-type-filter">
              <option value="all">All types</option>
              <option value="rgb">RGB</option>
              <option value="xray">X-ray</option>
              <option value="adaptive">Adaptive</option>
            </select>
          </div>
          <button
            className="button button-ghost button-sm"
            onClick={() => setSortDescending((c) => !c)}
            style={{ marginBottom: '2px' }}
          >
            <ArrowUpDown size={14} />
            {sortDescending ? 'Newest first' : 'Oldest first'}
          </button>
        </div>
      </div>

      {/* ── Table ─────────────────────────────────────────────── */}
      {rows.length === 0 ? (
        <div className="panel panel-pad empty-state">
          <HistoryIcon size={36} strokeWidth={1.2} className="empty-state-icon" />
          <p>No inspections match the current filters.</p>
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Type</th>
                <th>Status</th>
                <th>Prediction</th>
                <th>Score</th>
                <th>Created</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((item) => (
                <tr key={item.id}>
                  <td>
                    <code style={{ fontSize: '0.78rem', color: 'var(--primary)', fontWeight: 500 }}>
                      {item.id.slice(0, 8)}
                    </code>
                  </td>
                  <td>
                    <span className="badge badge-primary">{item.inspection_type}</span>
                  </td>
                  <td>
                    <span className={`status-pill ${item.status === 'completed' ? 'status-pass' : 'status-review'}`}>
                      {item.status}
                    </span>
                  </td>
                  <td style={{ fontSize: '0.85rem' }}>{item.predicted_class || '—'}</td>
                  <td style={{ fontSize: '0.85rem', fontFamily: 'monospace' }}>
                    {item.score === null ? '—' : item.score.toFixed(4)}
                  </td>
                  <td style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    {item.created_at ? new Date(item.created_at).toLocaleString() : '—'}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <div style={{ display: 'flex', gap: '0.35rem', justifyContent: 'flex-end' }}>
                      <button
                        className="button button-ghost button-sm"
                        onClick={() => navigate(`/reports?id=${item.id}`)}
                        title="View report"
                      >
                        <ExternalLink size={13} /> Details
                      </button>
                      <button
                        className="button button-danger button-sm"
                        onClick={() => remove(item.id)}
                        title="Delete inspection"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="muted" style={{ marginTop: '0.75rem', fontSize: '0.82rem' }}>
        {rows.length} displayed from {all.length} stored inspections.
      </p>
    </div>
  )
}
