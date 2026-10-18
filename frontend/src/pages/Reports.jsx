import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { listInspections, getReport } from '../api'
import {
  FileText,
  Download,
  Loader2,
  CheckCircle,
  AlertTriangle,
  Search,
  Copy,
  Check,
  ExternalLink,
  Eye,
  Cpu,
  Smartphone,
  User,
} from 'lucide-react'

export default function Reports() {
  const location = useLocation()
  const navigate = useNavigate()
  const [inspections, setInspections] = useState([])
  const [searchTerm, setSearchTerm] = useState('')
  const [selected, setSelected] = useState('')
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [listError, setListError] = useState(null)
  const [reportError, setReportError] = useState(null)
  const [copied, setCopied] = useState(false)
  const [downloadSuccess, setDownloadSuccess] = useState('')

  useEffect(() => {
    let active = true
    const searchParams = new URLSearchParams(location.search)
    const urlId = location.state?.inspectionId || searchParams.get('id')

    const load = async () => {
      try {
        const data = await listInspections({ limit: 100 })
        if (!active) return
        const items = data.items || []
        setInspections(items)

        const targetId = urlId || (items.length ? items[0].id : '')
        if (targetId) {
          setSelected(targetId)
          loadSingleReport(targetId)
        }
      } catch (err) {
        if (active) setListError(err.message)
      }
    }
    load()
    return () => {
      active = false
    }
  }, [location.search, location.state])

  async function loadSingleReport(id) {
    if (!id) {
      setReport(null)
      return
    }
    setLoading(true)
    setReportError(null)
    try {
      const data = await getReport(id, 'json')
      setReport(data)
    } catch (err) {
      setReportError(err.message)
      setReport(null)
    } finally {
      setLoading(false)
    }
  }

  function handleSelectChange(id) {
    setSelected(id)
    if (id) {
      navigate(`/reports?id=${id}`, { replace: true })
      loadSingleReport(id)
    } else {
      setReport(null)
    }
  }

  const filteredInspections = inspections.filter((item) => {
    if (!searchTerm.trim()) return true
    const term = searchTerm.toLowerCase()
    const idMatch = (item.id || '').toLowerCase().includes(term)
    const fileMatch = (item.original_filename || '').toLowerCase().includes(term)
    const typeMatch = (item.inspection_type || '').toLowerCase().includes(term)
    const objMatch = (item.detected_object || '').toLowerCase().includes(term)
    const condMatch = (item.condition_label || '').toLowerCase().includes(term)
    return idMatch || fileMatch || typeMatch || objMatch || condMatch
  })

  // Determine clean name for downloaded file
  const selectedItem = inspections.find((i) => i.id === selected)
  const rawBaseName = selectedItem?.original_filename
    ? selectedItem.original_filename.replace(/\.[^/.]+$/, '')
    : selected ? selected.slice(0, 8) : 'report'
  const cleanBaseName = rawBaseName.replace(/[^a-zA-Z0-9_-]/g, '_')

  function triggerDownloadNotification(msg) {
    setDownloadSuccess(msg)
    setTimeout(() => setDownloadSuccess(''), 4000)
  }

  function downloadJson() {
    if (!report) return
    const blob = new Blob([JSON.stringify(report.report || report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `report_${cleanBaseName}_${report.inspection_id.slice(0, 8)}.json`
    link.click()
    URL.revokeObjectURL(url)
    triggerDownloadNotification(`Downloaded JSON report (${link.download})`)
  }

  function downloadMarkdown() {
    if (!selected) return
    setLoading(true)
    setReportError(null)
    getReport(selected, 'markdown')
      .then((markdown) => {
        const blob = new Blob([markdown], { type: 'text/markdown' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `report_${cleanBaseName}_${selected.slice(0, 8)}.md`
        link.click()
        URL.revokeObjectURL(url)
        triggerDownloadNotification(`Downloaded Markdown report (${link.download})`)
      })
      .catch((err) => {
        setReportError(err.message)
      })
      .finally(() => {
        setLoading(false)
      })
  }

  function downloadHtml() {
    if (!selected) return
    setLoading(true)
    setReportError(null)
    getReport(selected, 'html')
      .then((htmlContent) => {
        const blob = new Blob([htmlContent], { type: 'text/html' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `report_${cleanBaseName}_${selected.slice(0, 8)}.html`
        link.click()
        URL.revokeObjectURL(url)
        triggerDownloadNotification(`Downloaded HTML printable report (${link.download})`)
      })
      .catch((err) => {
        setReportError(err.message)
      })
      .finally(() => {
        setLoading(false)
      })
  }

  function copyId() {
    if (!selected) return
    navigator.clipboard.writeText(selected)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }



  const repData = report?.report || {}
  const pred = repData.prediction || {}
  const detectedSubject = pred.detected_object || selectedItem?.detected_object || 'Printed Circuit Board (PCB)'
  const condition = pred.condition_label || selectedItem?.condition_label || (pred.predicted_class === 'defective' ? 'Defective' : 'Normal')
  const isDefective = pred.is_defective || pred.predicted_class === 'defective' || selectedItem?.is_defective

  const scoreVal = pred.score ?? repData.rgb_metrics?.anomaly_score
  const scoreDisplay = typeof scoreVal === 'number' ? scoreVal.toFixed(4) : (scoreVal || 'N/A')

  const threshVal = pred.threshold ?? repData.rgb_metrics?.threshold ?? repData.rgb_metrics?.localization?.image_threshold ?? 3.3498
  const threshDisplay = typeof threshVal === 'number' ? threshVal.toFixed(4) : (threshVal || 'N/A')

  const storedImgRef = repData.stored_image_reference || selectedItem?.stored_image_reference

  return (
    <div className="page-shell">
      <div className="page-header">
        <h1>Inspection Report Center</h1>
        <p>Review comprehensive inspection reports, inspect telemetry, and download formatted reports by ID or file name.</p>
      </div>

      {downloadSuccess && (
        <div
          style={{
            marginBottom: '1rem',
            padding: '0.75rem 1rem',
            background: 'rgba(16, 185, 129, 0.12)',
            border: '1px solid rgba(16, 185, 129, 0.35)',
            borderRadius: 'var(--radius-sm)',
            color: '#10b981',
            fontWeight: 600,
            fontSize: '0.84rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <CheckCircle size={16} /> {downloadSuccess}
        </div>
      )}

      <div className="grid-2">
        {/* ── Left Column: Search, Select & Actions ─────────────── */}
        <div className="panel panel-pad">
          <div className="heading-md" style={{ marginBottom: '0.35rem' }}>Select Inspection</div>
          <p className="muted" style={{ fontSize: '0.78rem', marginBottom: '1rem' }}>
            Filter and pick any inspection record by ID, filename, or subject name.
          </p>

          {/* Search Filter Box */}
          <div style={{ position: 'relative', marginBottom: '0.75rem' }}>
            <Search size={15} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search by ID, file name, or object..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: '100%',
                paddingLeft: '2rem',
                fontSize: '0.84rem',
                boxSizing: 'border-box',
              }}
            />
          </div>

          {/* Select Dropdown */}
          <div style={{ marginBottom: '1.25rem' }}>
            <div className="stat-label" style={{ marginBottom: '0.35rem' }}>
              Inspection Records ({filteredInspections.length} available)
            </div>
            <select
              value={selected}
              onChange={(e) => handleSelectChange(e.target.value)}
              id="report-inspection-select"
              style={{ fontSize: '0.84rem', width: '100%' }}
            >
              <option value="">-- Choose an inspection record --</option>
              {filteredInspections.map((item) => {
                const label = item.original_filename || item.inspection_type
                const shortId = item.id.slice(0, 8)
                const dateStr = item.created_at ? new Date(item.created_at).toLocaleDateString() : ''
                const subj = item.detected_object ? ` • ${item.detected_object}` : ''
                return (
                  <option key={item.id} value={item.id}>
                    {shortId} — {label}{subj} ({dateStr})
                  </option>
                )
              })}
            </select>
          </div>

          {selected && (
            <div style={{ padding: '0.75rem', background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 650, color: 'var(--text-soft)' }}>Selected Inspection ID:</span>
                <button
                  className="button button-secondary button-sm"
                  style={{ padding: '0.15rem 0.45rem', fontSize: '0.72rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                  onClick={copyId}
                >
                  {copied ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
                  {copied ? 'Copied' : 'Copy ID'}
                </button>
              </div>
              <code style={{ fontSize: '0.78rem', wordBreak: 'break-all', color: 'var(--primary)', fontWeight: 650 }}>
                {selected}
              </code>
            </div>
          )}

          <div className="section-title" style={{ fontSize: '0.84rem', marginBottom: '0.5rem' }}>Export & Download Reports</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <button
              className="button button-primary"
              disabled={!selected || loading}
              onClick={downloadMarkdown}
              id="download-markdown"
            >
              <Download size={15} /> Download Markdown (.md)
            </button>
            <button
              className="button button-secondary"
              disabled={!selected || loading}
              onClick={downloadJson}
              id="download-json"
            >
              <Download size={15} /> Download JSON Data (.json)
            </button>
            <button
              className="button button-secondary"
              disabled={!selected || loading}
              onClick={downloadHtml}
              id="download-html"
            >
              <Download size={15} /> Download Printable HTML (.html)
            </button>
          </div>

          {reportError && (
            <div style={{ marginTop: '0.85rem', padding: '0.6rem 0.85rem', background: 'var(--danger-soft)', border: '1px solid rgba(220,38,38,0.15)', borderRadius: 'var(--radius-sm)', fontSize: '0.82rem', color: 'var(--danger)' }}>
              {reportError}
            </div>
          )}
        </div>

        {/* ── Right Column: Report Telemetry Preview ────────────── */}
        <div className="panel panel-pad">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <div className="heading-md" style={{ margin: 0 }}>Inspection Report Preview</div>
            {report && (
              <span className={`status-pill ${isDefective ? 'status-review' : 'status-pass'}`}>
                {isDefective ? 'DEFECTIVE' : 'PASS'}
              </span>
            )}
          </div>

          {!report ? (
            <div className="empty-state">
              <FileText size={36} strokeWidth={1.2} className="empty-state-icon" />
              <p>{loading ? 'Loading inspection report...' : 'Select an inspection on the left to preview its full report.'}</p>
            </div>
          ) : (
            <>
              {/* Optional Inspected Image Preview */}
              {storedImgRef && (
                <div style={{ marginBottom: '1rem', borderRadius: 'var(--radius-sm)', overflow: 'hidden', border: '1px solid var(--border)' }}>
                  <div style={{ padding: '0.45rem 0.75rem', background: 'var(--surface-2)', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.78rem', fontWeight: 650, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <Eye size={13} /> Inspected Image
                    </span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      {repData.original_filename || selectedItem?.original_filename || 'source'}
                    </span>
                  </div>
                  <div style={{ background: '#080d1a', display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '0.5rem', maxHeight: '200px' }}>
                    <img
                      src={`http://127.0.0.1:8000/api/storage/inspection_images/${storedImgRef}`}
                      alt="Stored inspection"
                      style={{ maxHeight: '180px', maxWidth: '100%', objectFit: 'contain', borderRadius: 4 }}
                      onError={(e) => {
                        e.target.style.display = 'none'
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Metrics Grid */}
              <div style={{ display: 'grid', gap: '0.75rem' }}>
                <ReportField label="Target Subject" value={<strong>{detectedSubject}</strong>} />
                <ReportField
                  label="Condition Assessment"
                  value={
                    <span style={{ color: isDefective ? '#ef4444' : '#10b981', fontWeight: 650 }}>
                      {condition}
                    </span>
                  }
                />
                <ReportField label="Anomaly Score" value={scoreDisplay} />
                <ReportField label="Dynamic Threshold" value={threshDisplay} />
                <ReportField label="Original Filename" value={repData.original_filename || selectedItem?.original_filename || '—'} />
                <ReportField label="Inspection Type" value={(repData.inspection_type || 'rgb').toUpperCase()} />
                <ReportField label="Timestamp" value={repData.created_at ? new Date(repData.created_at).toLocaleString() : '—'} />
                <ReportField label="Engine Model ID" value={repData.model?.model_id || 'rgb-pixel-mahalanobis-anomaly'} />
              </div>

              {/* Decision reason & notes */}
              <div style={{ marginTop: '1rem', padding: '0.85rem 1rem', background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                <div className="stat-label">Analysis Findings</div>
                <p style={{ margin: '0.25rem 0 0', fontSize: '0.82rem', color: 'var(--text-main)', lineHeight: 1.45 }}>
                  {repData.rgb_metrics?.message || repData.adaptive_decision?.reason || (
                    isDefective
                      ? 'Defect/breakout damage was detected exceeding the dynamic anomaly threshold.'
                      : 'Subject evaluated within normal operating tolerances. No critical anomalies found.'
                  )}
                </p>
              </div>

              {/* Action Bar */}
              <div style={{ marginTop: '1.25rem', display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
                <button className="button button-primary" onClick={downloadMarkdown}>
                  <Download size={14} /> Download .md
                </button>
                <button className="button button-secondary" onClick={downloadJson}>
                  <Download size={14} /> Download .json
                </button>
                <button className="button button-secondary" onClick={downloadHtml}>
                  <Download size={14} /> Download .html
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function ReportField({ label, value }) {
  return (
    <div className="metric-line">
      <span className="muted" style={{ fontSize: '0.82rem' }}>{label}</span>
      <span style={{ fontSize: '0.875rem' }}>{value}</span>
    </div>
  )
}
