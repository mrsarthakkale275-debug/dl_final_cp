import { useEffect, useState } from 'react'
import { getHealth, getDatasetStatus, getModelStatus } from '../api'
import { Server, Database, Cpu, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'

export default function ModelStatus() {
  const [data, setData] = useState({ health: null, models: null, datasets: null })
  const [error, setError] = useState(null)

  useEffect(() => {
    let active = true
    async function load() {
      try {
        const [health, models, datasets] = await Promise.all([getHealth(), getModelStatus(), getDatasetStatus()])
        if (!active) return
        setData({ health, models, datasets })
      } catch (err) {
        if (active) setError(err.message)
      }
    }
    load()
    return () => { active = false }
  }, [])

  if (error) {
    return (
      <div className="page-shell">
        <div className="panel panel-pad">
          <p style={{ color: 'var(--danger)' }}>{error}</p>
          <button className="button button-secondary" style={{ marginTop: '0.75rem' }} onClick={() => window.location.reload()}>Retry</button>
        </div>
      </div>
    )
  }

  if (!data.models || !data.datasets) {
    return (
      <div className="page-shell">
        <div className="spinner-center"><div className="spinner" /></div>
      </div>
    )
  }

  const { models, health, datasets } = data

  return (
    <div className="page-shell">
      <div className="page-header">
        <h1>Model Status</h1>
        <p>Infrastructure health, model readiness, and dataset summary.</p>
      </div>

      {/* ── Status cards ──────────────────────────────────────── */}
      <div className="grid-4">
        <StatusCard
          icon={<Server size={18} />}
          title="API"
          status={health?.status || 'Unknown'}
          detail="FastAPI application availability."
          ready={health?.status === 'healthy'}
        />
        <StatusCard
          icon={<Database size={18} />}
          title="Database"
          status={health?.database || 'Unknown'}
          detail="SQLite/Supabase connection check."
          ready={health?.database === 'available'}
        />
        <StatusCard
          icon={<Cpu size={18} />}
          title="RGB Model"
          status={models.rgb.status}
          detail={models.rgb.message}
          ready={models.rgb.ready}
        />
        <StatusCard
          icon={<AlertTriangle size={18} />}
          title="X-ray Model"
          status={models.xray.status}
          detail={models.xray.message}
          ready={models.xray.ready}
        />
      </div>

      {/* ── Model details ─────────────────────────────────────── */}
      <div className="section grid-2">
        <div className="panel panel-pad">
          <div className="heading-md" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircle size={16} style={{ color: models.rgb.ready ? 'var(--success)' : 'var(--text-soft)' }} />
            RGB Model
          </div>
          <p className="muted" style={{ marginBottom: '1rem', fontSize: '0.85rem' }}>{models.rgb.message}</p>

          <MetricLine label="Model ID" value={models.rgb.model_id || 'Not available'} />
          <MetricLine label="Version" value={models.rgb.version || 'Not available'} />
          <MetricLine label="Weights" value={models.rgb.weights.length ? models.rgb.weights.join(', ') : 'None'} />

          <div style={{ marginTop: '1rem', padding: '0.85rem 1rem', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
            <div className="section-title" style={{ margin: '0 0 0.3rem' }}>Evaluation</div>
            <p className="muted" style={{ margin: 0, fontSize: '0.82rem' }}>
              Independently measured metrics are stored in <code style={{ background: 'rgba(0,0,0,0.04)', padding: '0.1rem 0.25rem', borderRadius: 3, fontSize: '0.78rem' }}>models/rgb/anomaly_evaluation.json</code>.
              Results are intentionally not overstated.
            </p>
          </div>

          <span className="badge badge-review" style={{ marginTop: '0.85rem' }}>
            Limited performance — not production-ready
          </span>
        </div>

        <div className="panel panel-pad">
          <div className="heading-md" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <XCircle size={16} style={{ color: 'var(--danger)' }} />
            X-ray Model
          </div>
          <p className="muted" style={{ marginBottom: '1rem', fontSize: '0.85rem' }}>{models.xray.message}</p>

          <MetricLine label="Model ID" value={models.xray.model_id || 'Not available'} />
          <MetricLine label="Version" value={models.xray.version || 'Not available'} />
          <MetricLine label="Weights" value={models.xray.weights.length ? models.xray.weights.join(', ') : 'None'} />

          <span className="badge badge-unavailable" style={{ marginTop: '0.85rem' }}>
            X-ray model unavailable — human review required
          </span>
        </div>
      </div>

      {/* ── Dataset summary ───────────────────────────────────── */}
      <div className="section">
        <div className="panel panel-pad">
          <div className="heading-md">Dataset Summary</div>
          <div className="grid-4">
            <DatasetCard label="RGB train records" value={datasets.rgb.train_records} color="var(--primary)" bg="var(--primary-soft)" />
            <DatasetCard label="RGB test records" value={datasets.rgb.test_records} color="var(--cyan)" bg="var(--cyan-soft)" />
            <DatasetCard label="X-ray records" value={datasets.xray.xray_records} color="var(--warning)" bg="var(--warning-soft)" />
            <DatasetCard label="Excluded records" value={datasets.xray.excluded_records} color="var(--text-muted)" bg="var(--surface)" />
          </div>
        </div>
      </div>
    </div>
  )
}

function StatusCard({ icon, title, status, detail, ready }) {
  return (
    <div className="panel stat-card">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: ready ? 'var(--success-soft)' : 'var(--surface)',
          color: ready ? 'var(--success)' : 'var(--text-soft)',
          display: 'grid', placeItems: 'center',
        }}>
          {icon}
        </div>
        <span className={`status-dot ${ready ? 'ready' : 'not-ready'}`} />
      </div>
      <div style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '0.15rem' }}>{title}</div>
      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: ready ? 'var(--success)' : 'var(--text-muted)', marginBottom: '0.3rem' }}>{status}</div>
      <p className="muted" style={{ margin: 0, fontSize: '0.78rem', lineHeight: 1.45 }}>{detail}</p>
    </div>
  )
}

function MetricLine({ label, value }) {
  return (
    <div className="metric-line">
      <span className="muted">{label}</span>
      <strong style={{ fontSize: '0.875rem' }}>{String(value)}</strong>
    </div>
  )
}

function DatasetCard({ label, value, color, bg }) {
  return (
    <div style={{
      padding: '1rem 1.15rem',
      background: bg,
      borderRadius: 'var(--radius-sm)',
      border: '1px solid var(--border)',
    }}>
      <div className="stat-label">{label}</div>
      <div style={{ fontSize: '1.5rem', fontWeight: 750, color, letterSpacing: '-0.03em' }}>{value}</div>
    </div>
  )
}
