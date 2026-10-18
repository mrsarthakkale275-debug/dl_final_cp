import { useNavigate } from 'react-router-dom'
import { ArrowRight, Upload, Cpu, Search, FileCheck, Database, Shield } from 'lucide-react'

export default function About() {
  const navigate = useNavigate()

  return (
    <div className="page-shell">
      <div className="page-header">
        <h1>About AdaptiveInspect AI</h1>
        <p>Platform overview, architecture, capabilities, and known limitations.</p>
      </div>

      {/* ── Overview ──────────────────────────────────────────── */}
      <div className="panel panel-pad" style={{ maxWidth: 740 }}>
        <div className="heading-md">Purpose</div>
        <p className="muted" style={{ marginBottom: '0.85rem' }}>
          AdaptiveInspect AI is a final-year engineering project for PCB quality inspection. It
          separates RGB anomaly analysis from X-ray review, stores inspection history, and uses
          policy-level adaptive recommendations with explicit reasons and human-review safety.
        </p>
        <p className="muted" style={{ marginBottom: '0.85rem' }}>
          The RGB model is a normal-only per-pixel statistical anomaly detector. Its evaluation
          metrics are independently measured and intentionally not overstated. The X-ray dataset
          has no verified normal/no-defect ground truth currently, so exact X-ray inference is
          disabled and requests are routed to human review.
        </p>
        <p className="muted" style={{ margin: 0 }}>
          RGB and X-ray datasets are not paired images of the same physical PCB. The adaptive policy
          combines available results without claiming multimodal feature fusion.
        </p>
      </div>

      {/* ── Architecture & Status ─────────────────────────────── */}
      <div className="section grid-2">
        <div className="panel panel-pad">
          <div className="heading-md">Implementation Status</div>
          <div style={{ display: 'grid', gap: '0.5rem' }}>
            <StatusItem
              icon={<Cpu size={16} />}
              text="FastAPI backend connected and available."
              ready
            />
            <StatusItem
              icon={<Shield size={16} />}
              text="RGB model integrated through the model registry."
              ready
            />
            <StatusItem
              icon={<Search size={16} />}
              text="X-ray model intentionally unavailable pending suitable data."
              ready={false}
            />
            <StatusItem
              icon={<Database size={16} />}
              text="PostgreSQL/Supabase persistence supported through environment configuration."
              ready
            />
          </div>
        </div>

        <div className="panel panel-pad">
          <div className="heading-md">Architecture Flow</div>
          <div style={{ display: 'grid', gap: '0.5rem' }}>
            {[
              { icon: <Upload size={15} />, label: 'Image upload and validation' },
              { icon: <Cpu size={15} />, label: 'RGB anomaly model' },
              { icon: <Shield size={15} />, label: 'Adaptive decision policy' },
              { icon: <Database size={15} />, label: 'Inspection history and metadata' },
              { icon: <FileCheck size={15} />, label: 'Reports and dashboard' },
            ].map((step, index) => (
              <div
                key={step.label}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  padding: '0.7rem 0.85rem',
                  background: 'var(--surface-2)',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)',
                }}
              >
                <div style={{
                  width: 30, height: 30,
                  borderRadius: 8,
                  background: 'var(--primary-soft)',
                  color: 'var(--primary)',
                  display: 'grid',
                  placeItems: 'center',
                  flexShrink: 0,
                  fontSize: '0.78rem',
                  fontWeight: 700,
                }}>
                  {index + 1}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                  {step.icon}
                  {step.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Limitations ───────────────────────────────────────── */}
      <div className="section">
        <div className="panel panel-pad" style={{ maxWidth: 740 }}>
          <div className="heading-md">Known Limitations</div>
          <div style={{ display: 'grid', gap: '0.4rem' }}>
            <LimitItem text="The RGB anomaly model has limited performance and is not production-ready." />
            <LimitItem text="X-ray void detection is disabled because no verified ground-truth dataset exists." />
            <LimitItem text="No genuine multimodal RGB/X-ray feature fusion is implemented." />
            <LimitItem text="Anomaly localization is statistical estimation, not verified ground-truth segmentation." />
            <LimitItem text="No external certifications or independent audits have been conducted." />
          </div>
        </div>
      </div>

      {/* ── CTA ───────────────────────────────────────────────── */}
      <div className="section" style={{ marginBottom: '2rem' }}>
        <div style={{
          background: 'linear-gradient(135deg, var(--primary-soft), var(--surface-2))',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          flexWrap: 'wrap',
        }}>
          <div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.25rem' }}>
              Try the inspection workflow
            </div>
            <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>
              Upload a PCB image and see the analysis pipeline in action.
            </p>
          </div>
          <button className="button button-primary" onClick={() => navigate('/inspection')}>
            Start Inspection <ArrowRight size={15} />
          </button>
        </div>
      </div>
    </div>
  )
}

function StatusItem({ icon, text, ready }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.65rem',
      padding: '0.65rem 0.85rem',
      background: ready ? 'var(--success-soft)' : 'var(--warning-soft)',
      borderRadius: 'var(--radius-sm)',
      border: `1px solid ${ready ? 'rgba(21,128,61,0.12)' : 'rgba(180,83,9,0.12)'}`,
    }}>
      <span className={`status-dot ${ready ? 'ready' : 'not-ready'}`} />
      <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{text}</span>
    </div>
  )
}

function LimitItem({ text }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: '0.5rem',
      padding: '0.5rem 0',
      borderBottom: '1px solid var(--border)',
      fontSize: '0.85rem',
      color: 'var(--text-muted)',
    }}>
      <span style={{ color: 'var(--warning)', flexShrink: 0, marginTop: '0.1rem' }}>•</span>
      {text}
    </div>
  )
}
