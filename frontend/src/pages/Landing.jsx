import { lazy, Suspense, useEffect, useState } from 'react'
import SceneErrorBoundary from '../components/SceneErrorBoundary.jsx'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, Cpu, Upload, Search, FileCheck, BarChart3, Shield } from 'lucide-react'

const HeroScene = lazy(() => import('../components/HeroScene.jsx'))

function supportsWebGL() {
  if (typeof document === 'undefined') return false
  try {
    const c = document.createElement('canvas')
    return Boolean(c.getContext('webgl') || c.getContext('experimental-webgl') || c.getContext('webgl2'))
  } catch { return false }
}

function reducedMotion() {
  return typeof window !== 'undefined' && (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false)
}

export default function Landing() {
  const navigate = useNavigate()
  const [useScene, setUseScene] = useState(false)

  useEffect(() => {
    setUseScene(supportsWebGL() && !reducedMotion())
  }, [])

  return (
    <main className="landing">
      {/* ── Full-screen 3D background ─────────────────────────────── */}
      <div className="landing-bg" aria-hidden="true">
        {useScene ? (
          <SceneErrorBoundary fallback={<SceneFallback />}>
            <Suspense fallback={<SceneFallback />}>
              <HeroScene />
            </Suspense>
          </SceneErrorBoundary>
        ) : (
          <SceneFallback />
        )}
      </div>

      {/* ── Navigation ────────────────────────────────────────────── */}
      <header className="landing-nav">
        <div className="brand" onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>
          <div className="brand-mark">AI</div>
          <div>
            <div className="brand-name">AdaptiveInspect AI</div>
            <div className="brand-subtitle">PCB quality inspection platform</div>
          </div>
        </div>
        <nav className="landing-nav-links" aria-label="Landing navigation">
          <button className="button button-ghost button-sm" onClick={() => navigate('/overview')} style={{ color: 'var(--text-main)' }}>
            Overview
          </button>
          <button className="button button-ghost button-sm" onClick={() => navigate('/history')} style={{ color: 'var(--text-main)' }}>
            History
          </button>
          <button className="button button-ghost button-sm" onClick={() => navigate('/reports')} style={{ color: 'var(--text-main)' }}>
            Reports
          </button>
          <button className="button button-primary button-sm" onClick={() => navigate('/inspection')}>
            New Inspection
          </button>
        </nav>
      </header>

      {/* ── Hero content (layered over 3D background) ───────────── */}
      <section className="landing-hero" aria-label="Hero">
        <div className="hero-content">
          <div className="landing-eyebrow">
            INTELLIGENT PCB QUALITY INSPECTION
          </div>

          <h1 className="landing-title">
            Precision Inspection.<br />Smarter Electronics.
          </h1>

          <p className="landing-subtitle">
            Analyze PCB images with AI-assisted anomaly detection, visualize
            suspicious regions, and review inspection results through one
            streamlined workspace.
          </p>

          <div className="hero-actions">
            <button className="button button-primary" onClick={() => navigate('/inspection')}>
              Start an Inspection
            </button>
            <button className="button button-secondary" onClick={() => navigate('/overview')}>
              Explore the Platform
              <ArrowRight size={16} />
            </button>
          </div>

          <p className="hero-tech-line">
            RGB anomaly analysis · Visual results · Inspection history
          </p>
        </div>
      </section>

      {/* ── Workflow steps ────────────────────────────────────────── */}
      <section id="platform" className="workflow-section" aria-label="How it works">
        <h2 className="workflow-section-title">How It Works</h2>
        <p className="workflow-section-sub">
          Three clear stages from upload to actionable results.
        </p>

        <div className="workflow-steps">
          <div className="panel workflow-step">
            <div className="workflow-step-number">
              <Upload size={18} />
            </div>
            <h3>Upload</h3>
            <p>
              Select a PCB image from your file system or capture one with
              your camera. Supported formats include JPEG, PNG, BMP, TIFF,
              and WebP.
            </p>
          </div>

          <div className="panel workflow-step">
            <div className="workflow-step-number">
              <Search size={18} />
            </div>
            <h3>Analyze</h3>
            <p>
              The backend inspects the image using the RGB anomaly model,
              computes anomaly scores, and applies the adaptive decision
              policy.
            </p>
          </div>

          <div className="panel workflow-step">
            <div className="workflow-step-number">
              <FileCheck size={18} />
            </div>
            <h3>Review</h3>
            <p>
              View the outcome, localization metadata, and recommendations.
              Download reports, browse history, and track all inspections.
            </p>
          </div>
        </div>
      </section>

      {/* ── Capabilities ──────────────────────────────────────────── */}
      <section id="capabilities" className="feature-section" aria-label="Capabilities">
        <div className="feature-strip">
          <div className="panel feature-tile">
            <div className="feature-icon">
              <Cpu size={20} />
            </div>
            <h3>RGB Inspection</h3>
            <p>
              Upload a real PCB image and review anomaly score, status,
              localization metadata, and generated reports.
            </p>
          </div>

          <div className="panel feature-tile">
            <div className="feature-icon">
              <Shield size={20} />
            </div>
            <h3>Adaptive Decisions</h3>
            <p>
              Policy-level recommendations combine available RGB results with
              explicit reasons and human-review safety.
            </p>
          </div>

          <div className="panel feature-tile">
            <div className="feature-icon">
              <BarChart3 size={20} />
            </div>
            <h3>Dashboard Analytics</h3>
            <p>
              Backend-driven counts for inspections, RGB results, adaptive
              outcomes, and recent activity.
            </p>
          </div>

          <div className="panel feature-tile">
            <div className="feature-icon">
              <FileCheck size={20} />
            </div>
            <h3>Honest Status</h3>
            <p>
              Missing models are shown as unavailable. No fake confidence or
              unverified X-ray predictions.
            </p>
          </div>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────────────── */}
      <section className="cta-section" aria-label="Call to action">
        <div className="cta-band">
          <h2>Ready to Inspect?</h2>
          <p>
            Upload your first PCB image and see the inspection workflow in action.
          </p>
          <button
            className="button button-primary"
            onClick={() => navigate('/inspection')}
            id="cta-start-inspection"
          >
            Start an Inspection
            <ArrowRight size={16} />
          </button>
        </div>
      </section>
    </main>
  )
}

function SceneFallback() {
  return (
    <div className="scene-fallback">
      <Cpu size={36} strokeWidth={1.2} />
      <span>3D visualization unavailable</span>
    </div>
  )
}
