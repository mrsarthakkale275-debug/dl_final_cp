import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Camera, Loader2, Trash2, Upload, Image, ArrowRight, AlertTriangle, Eye, CheckCircle } from 'lucide-react'
import { inspectImage } from '../api'

export default function InspectionWorkspace() {
  const navigate = useNavigate()
  const webcamVideoRef = useRef(null)
  const webcamCanvasRef = useRef(null)
  const streamRef = useRef(null)
  const [mode, setMode] = useState('upload')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [webcamActive, setWebcamActive] = useState(false)
  const [capturedBlob, setCapturedBlob] = useState(null)
  const [endpoint, setEndpoint] = useState('rgb')
  const [submitting, setSubmitting] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  function setPickedFile(nextFile) {
    if (!nextFile) { setFile(null); setPreviewUrl(null); return }
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp']
    if (!allowedTypes.includes(nextFile.type) || nextFile.size > 10 * 1024 * 1024) {
      setError('Choose a supported image under 10 MB.')
      return
    }
    setError(null)
    setFile(nextFile)
    setPreviewUrl(URL.createObjectURL(nextFile))
    setCapturedBlob(null)
  }

  async function startWebcam() {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser does not support camera access.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      streamRef.current = stream
      setWebcamActive(true)
      setTimeout(() => {
        if (webcamVideoRef.current) webcamVideoRef.current.srcObject = stream
      }, 0)
    } catch (err) {
      setError(err?.message || 'Camera permission was not granted.')
    }
  }

  function stopWebcam() {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setWebcamActive(false)
  }

  function captureWebcam() {
    const video = webcamVideoRef.current
    const canvas = webcamCanvasRef.current
    if (!video || !canvas || video.videoWidth === 0) {
      setError('Wait for the camera preview before capturing.')
      return
    }
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (!blob) { setError('Could not capture the camera image.'); return }
      stopWebcam()
      setFile(new File([blob], 'webcam-capture.png', { type: 'image/png' }))
      setPreviewUrl(canvas.toDataURL('image/png'))
      setCapturedBlob(blob)
      setMode('upload')
    }, 'image/png')
  }

  async function submit() {
    if (!file && !capturedBlob) { setError('Select or capture an image before submitting.'); return }
    setSubmitting(true)
    setError(null)
    try {
      const submissionFile = capturedBlob
        ? new File([capturedBlob], 'webcam-capture.png', { type: 'image/png' })
        : file
      const data = await inspectImage(endpoint, submissionFile)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  function reset() {
    setFile(null)
    setPreviewUrl(null)
    setCapturedBlob(null)
    setError(null)
    setResult(null)
  }

  const fileSize = file ? (file.size / 1024).toFixed(1) + ' KB' : null

  return (
    <div className="page-shell">
      <div className="page-header">
        <h1>Inspection Workspace</h1>
        <p>Upload or capture a PCB image, choose an inspection mode, and review the results.</p>
      </div>

      {/* ── Controls bar ──────────────────────────────────────── */}
      <div className="surface-band" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
          <div>
            <div className="section-title" style={{ margin: '0 0 0.5rem' }}>Source</div>
            <div className="mode-toggle">
              <button
                className={`button ${mode === 'upload' ? 'active' : ''}`}
                onClick={() => setMode('upload')}
                id="mode-upload"
              >
                <Upload size={14} /> Upload
              </button>
              <button
                className={`button ${mode === 'webcam' ? 'active' : ''}`}
                onClick={() => { setMode('webcam'); startWebcam() }}
                id="mode-webcam"
              >
                <Camera size={14} /> Webcam
              </button>
            </div>
          </div>

          <div>
            <div className="section-title" style={{ margin: '0 0 0.5rem' }}>Analysis mode</div>
            <div className="endpoint-selector">
              <button
                className={`endpoint-chip ${endpoint === 'rgb' ? 'active' : ''}`}
                onClick={() => setEndpoint('rgb')}
                id="endpoint-rgb"
              >
                RGB
              </button>
              <button
                className={`endpoint-chip ${endpoint === 'xray' ? 'active' : ''}`}
                onClick={() => setEndpoint('xray')}
                id="endpoint-xray"
              >
                X-ray
              </button>
              <button
                className={`endpoint-chip ${endpoint === 'adaptive' ? 'active' : ''}`}
                onClick={() => setEndpoint('adaptive')}
                id="endpoint-adaptive"
              >
                Adaptive
              </button>
            </div>
          </div>
        </div>

        {endpoint === 'xray' && (
          <div className="warning-banner" style={{ marginTop: '1rem' }}>
            <strong><AlertTriangle size={13} style={{ display: 'inline', verticalAlign: '-2px' }} /> X-ray model unavailable — human review required</strong>
            <p>
              The backend does not have a trained X-ray void model. Any uploaded X-ray image will be
              handled by the safe review-required workflow, and no X-ray prediction will be displayed as fact.
            </p>
          </div>
        )}
      </div>

      {/* ── Main content ──────────────────────────────────────── */}
      <div className="grid-2">
        {/* Left: Upload / capture */}
        <div className="panel panel-pad">
          <div className="heading-md">
            {mode === 'upload' ? 'Upload image' : 'Camera capture'}
          </div>

          {mode === 'upload' ? (
            <>
              <label
                className={`drop-zone ${dragging ? 'active' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); setPickedFile(e.dataTransfer.files?.[0]) }}
                id="drop-zone"
              >
                <input
                  hidden
                  type="file"
                  accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp"
                  onChange={(e) => setPickedFile(e.target.files?.[0])}
                />
                <Upload size={28} className="drop-zone-icon" />
                <p style={{ margin: 0, fontWeight: 600, fontSize: '0.9rem' }}>
                  Drop an image here or click to browse
                </p>
                <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.8rem' }}>
                  JPG, PNG, BMP, TIFF, WEBP — up to 10 MB
                </p>
              </label>

              {previewUrl && (
                <div style={{ marginTop: '1rem' }}>
                  <div className="preview-frame">
                    <img className="preview-image" src={previewUrl} alt="Selected inspection preview" />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.65rem', gap: '0.5rem' }}>
                    <div>
                      <span style={{ fontSize: '0.82rem', fontWeight: 500 }}>{file?.name}</span>
                      {fileSize && <span className="muted" style={{ fontSize: '0.78rem', marginLeft: '0.4rem' }}>{fileSize}</span>}
                    </div>
                    <button className="button button-danger button-sm" onClick={reset}>
                      <Trash2 size={14} /> Remove
                    </button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div>
              <div className="preview-frame" style={{ minHeight: 280, background: '#0f172a' }}>
                {webcamActive ? (
                  <video ref={webcamVideoRef} className="preview-image" autoPlay playsInline muted />
                ) : (
                  <p style={{ color: 'var(--text-soft)', fontSize: '0.85rem' }}>Camera is inactive.</p>
                )}
              </div>
              <canvas ref={webcamCanvasRef} hidden />
              <div style={{ display: 'flex', gap: '0.6rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
                {!webcamActive ? (
                  <button className="button button-primary" onClick={startWebcam}>
                    <Camera size={15} /> Start Camera
                  </button>
                ) : (
                  <>
                    <button className="button button-primary" onClick={captureWebcam}>Capture Frame</button>
                    <button className="button button-danger" onClick={stopWebcam}>Cancel</button>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Submit / reset */}
          <div style={{ display: 'flex', gap: '0.6rem', marginTop: '1.25rem', flexWrap: 'wrap' }}>
            <button
              className="button button-primary"
              disabled={submitting || (!file && !capturedBlob)}
              onClick={submit}
              id="submit-inspection"
            >
              {submitting && <Loader2 size={15} style={{ animation: 'spin 700ms linear infinite' }} />}
              {submitting ? 'Analyzing...' : `Run ${endpoint.toUpperCase()} Inspection`}
            </button>
            <button className="button button-ghost" onClick={reset}>Reset</button>
          </div>

          {error && (
            <div style={{ marginTop: '0.75rem', padding: '0.6rem 0.85rem', background: 'var(--danger-soft)', border: '1px solid rgba(220,38,38,0.15)', borderRadius: 'var(--radius-sm)', fontSize: '0.82rem', color: 'var(--danger)' }}>
              {error}
            </div>
          )}
        </div>

        {/* Right: Result */}
        <div className="panel panel-pad">
          <div className="heading-md">Result</div>

          {!result && !submitting && (
            <div className="empty-state">
              <Image size={36} strokeWidth={1.2} className="empty-state-icon" />
              <p>Submit an image to see the real backend response here.</p>
            </div>
          )}

          {submitting && (
            <div className="empty-state" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
              <div className="spinner" />
              <p className="muted">Uploading and analyzing the image...</p>
            </div>
          )}

          {result && <InspectionResult result={result} navigate={navigate} />}
        </div>
      </div>
    </div>
  )
}

/* ── Inspection result display ───────────────────────────────────────── */
function InspectionResult({ result, navigate }) {
  const localization = result.rgb?.localization || result.xray?.localization
  const eligibility = result.rgb?.eligibility
  const decisionOutcome = result.rgb?.decision_outcome

  const detectedObject = result.detected_object || result.rgb?.detected_object || ''
  const isHuman = detectedObject.toLowerCase().includes('human') || result.entity_type === 'human'
  const isAnimal = detectedObject.toLowerCase().includes('animal') || result.entity_type === 'animal'
  const isPCB = detectedObject.toLowerCase().includes('pcb') || (result.status === 'completed' && !isHuman && !isAnimal)
  const isUnsupported = !isPCB
  const isDefective = Boolean(result.is_defective || result.rgb?.is_defective || result.predicted_class === 'defective')
  const conditionLabel = result.condition_label || result.rgb?.condition_label || (isDefective ? 'Defective' : 'Normal')
  
  let predictedClassDisplay = result.predicted_class || 'Not available'
  let predictedClassColor = 'inherit'
  if (isHuman) {
    predictedClassDisplay = 'HUMAN (PERSON DETECTED) - NORMAL'
    predictedClassColor = '#10b981'
  } else if (isAnimal) {
    predictedClassDisplay = `${detectedObject.toUpperCase()} - NORMAL`
    predictedClassColor = '#10b981'
  } else if (!isPCB) {
    predictedClassDisplay = `${(detectedObject || 'OBJECT').toUpperCase()} - ${isDefective ? 'DEFECTIVE (SURFACE DAMAGE)' : 'NORMAL (INTACT)'}`
    predictedClassColor = isDefective ? '#ef4444' : '#10b981'
  } else if (result.predicted_class === 'normal' || (!isDefective && isPCB)) {
    predictedClassDisplay = 'NORMAL PCB (PASS)'
    predictedClassColor = '#10b981'
  } else if (result.predicted_class === 'defective' || (isDefective && isPCB)) {
    predictedClassDisplay = 'DEFECTIVE PCB (BREAKOUT DETECTED)'
    predictedClassColor = '#ef4444'
  }

  const classScores = result.rgb?.class_scores
  const annotatedImage = result.annotated_image_base64 || result.qc_report?.annotated_image_base64
  const boxCount = localization?.bounding_boxes_xywh?.length ?? (isDefective ? 1 : 0)

  let outcomeBadge = 'badge-pass'
  let badgeText = 'PASS'
  if (isHuman) {
    outcomeBadge = 'badge-pass'
    badgeText = 'HUMAN (PERSON DETECTED)'
  } else if (isAnimal) {
    outcomeBadge = 'badge-pass'
    badgeText = detectedObject.toUpperCase()
  } else if (!isPCB) {
    outcomeBadge = isDefective ? 'badge-defect' : 'badge-pass'
    badgeText = `${(detectedObject || 'OBJECT').toUpperCase()} (${isDefective ? 'DEFECTIVE' : 'NORMAL'})`
  } else {
    outcomeBadge = isDefective ? 'badge-defect' : 'badge-pass'
    badgeText = isDefective ? 'PCB: DEFECTIVE (BREAKOUT)' : 'PCB: NORMAL (PASS)'
  }

  let statusDisplay = result.status
  if (isHuman) {
    statusDisplay = 'Human Detected (Living Subject)'
  } else if (isAnimal) {
    statusDisplay = 'Animal Detected (Living Subject)'
  } else if (!isPCB) {
    statusDisplay = isDefective ? 'Object Inspected (Defective)' : 'Object Inspected (Normal)'
  } else if (result.status === 'completed') {
    statusDisplay = isDefective ? 'PCB Completed (Defective)' : 'PCB Completed (Pass)'
  }

  const thresholdVal = localization?.image_threshold ?? result.threshold ?? result.rgb?.threshold ?? 3.3498
  const thresholdDisplay = typeof thresholdVal === 'number' ? thresholdVal.toFixed(4) : thresholdVal

  return (
    <>
      <div className="result-header">
        <span className="badge badge-primary">
          {result.inspection_type.toUpperCase()}
        </span>
        <span className={`badge ${outcomeBadge}`}>
          {badgeText}
        </span>
        {isPCB && eligibility && (
          <span className="badge badge-unsupported">
            {eligibility.toUpperCase().replace(/_/g, ' ')}
          </span>
        )}
      </div>

      {isHuman && (
        <div
          className="warning-banner"
          style={{
            marginTop: '0.85rem',
            background: 'rgba(16, 185, 129, 0.08)',
            borderColor: 'rgba(16, 185, 129, 0.35)',
          }}
        >
          <strong
            style={{
              color: '#10b981',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.88rem',
            }}
          >
            <CheckCircle size={15} /> Human Subject Detected: Person
          </strong>
          <p style={{ margin: '0.35rem 0 0', fontSize: '0.83rem', color: 'var(--text-main)' }}>
            Webcam input successfully recognized as a human person. Humans and living subjects are not classified as defective objects.
          </p>
        </div>
      )}

      {isAnimal && (
        <div
          className="warning-banner"
          style={{
            marginTop: '0.85rem',
            background: 'rgba(16, 185, 129, 0.08)',
            borderColor: 'rgba(16, 185, 129, 0.35)',
          }}
        >
          <strong
            style={{
              color: '#10b981',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.88rem',
            }}
          >
            <CheckCircle size={15} /> Animal Subject Detected: {detectedObject}
          </strong>
          <p style={{ margin: '0.35rem 0 0', fontSize: '0.83rem', color: 'var(--text-main)' }}>
            Input successfully recognized as an animal / pet. Living beings are not classified as defective objects.
          </p>
        </div>
      )}

      {!isPCB && !isHuman && !isAnimal && (
        <div
          className="warning-banner"
          style={{
            marginTop: '0.85rem',
            background: isDefective ? 'rgba(239, 68, 68, 0.08)' : 'rgba(16, 185, 129, 0.08)',
            borderColor: isDefective ? 'rgba(239, 68, 68, 0.35)' : 'rgba(16, 185, 129, 0.35)',
          }}
        >
          <strong
            style={{
              color: isDefective ? '#ef4444' : '#10b981',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.88rem',
            }}
          >
            <AlertTriangle size={15} />
            {isDefective ? `Defective Object Detected: ${detectedObject || 'Object'}` : `Normal Object Detected: ${detectedObject || 'Object'}`}
          </strong>
          <p style={{ margin: '0.35rem 0 0', fontSize: '0.83rem', color: 'var(--text-main)' }}>
            {result.rgb?.decision_reason || result.rgb?.message || result.adaptive?.reason || 'The object was evaluated with surface defect and fracture analysis.'}
          </p>
        </div>
      )}

      {result.stored_image_reference && (
        <p className="muted" style={{ fontSize: '0.78rem', marginBottom: '0.75rem', marginTop: !isPCB ? '0.75rem' : '0' }}>
          Stored reference: <code style={{ background: 'var(--surface-2)', padding: '0.1rem 0.3rem', borderRadius: 4, fontSize: '0.75rem' }}>{result.stored_image_reference}</code>
        </p>
      )}

      {/* Visual Defect / Breakout Overlay with RED bounding box */}
      {annotatedImage && (
        <div style={{ marginTop: '0.85rem', marginBottom: '1rem', borderRadius: 'var(--radius-sm)', overflow: 'hidden', border: '1px solid var(--border)' }}>
          <div style={{ padding: '0.55rem 0.85rem', background: 'var(--surface-2)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)', flexWrap: 'wrap', gap: '0.5rem' }}>
            <span style={{ fontWeight: 650, fontSize: '0.84rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <Eye size={15} /> Visual Defect Overlay
            </span>
            <span style={{ fontSize: '0.78rem', fontWeight: 650, color: isDefective ? '#ef4444' : '#10b981' }}>
              {isDefective ? `Defect / Breakout Box (Red Line: ${boxCount || 1})` : `Verified Normal (${detectedObject || 'Pass'})`}
            </span>
          </div>
          <div style={{ position: 'relative', background: '#0a0f1d', display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '0.75rem', minHeight: '220px' }}>
            <img
              src={annotatedImage.startsWith('data:') ? annotatedImage : `data:image/jpeg;base64,${annotatedImage}`}
              alt="Visual Defect Overlay with Red Line Box"
              style={{ maxWidth: '100%', maxHeight: '420px', objectFit: 'contain', borderRadius: '4px', boxShadow: '0 4px 14px rgba(0,0,0,0.45)' }}
            />
          </div>
          {isDefective && (
            <div style={{ padding: '0.45rem 0.85rem', background: 'rgba(239, 68, 68, 0.08)', borderTop: '1px solid rgba(239, 68, 68, 0.2)', fontSize: '0.78rem', color: '#ef4444', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <AlertTriangle size={14} /> Breakout / defect areas marked with red line box above.
            </div>
          )}
        </div>
      )}

      <div className="result-metrics">
        <div className="result-metric">
          <span className="stat-label">Predicted class</span>
          <strong style={{ color: predictedClassColor }}>{predictedClassDisplay}</strong>
        </div>
        <ResultMetric label="Status" value={statusDisplay} />
        <ResultMetric label="Anomaly score" value={result.score === null || result.score === undefined ? (result.anomaly_score !== undefined ? result.anomaly_score.toFixed(4) : 'Not available') : result.score.toFixed(4)} />
        <ResultMetric label="Threshold" value={thresholdDisplay} />
      </div>

      {isPCB && classScores && (
        <div style={{ marginTop: '0.75rem', padding: '0.6rem 0.85rem', background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', fontSize: '0.82rem' }}>
          <span style={{ fontWeight: 600, display: 'block', marginBottom: '0.2rem' }}>PCB Defect Confidence Breakdown</span>
          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <span>Normal PCB: <strong>{(classScores.normal * 100).toFixed(1)}%</strong></span>
            <span>Defective PCB: <strong>{(classScores.defective * 100).toFixed(1)}%</strong></span>
          </div>
        </div>
      )}

      {!isPCB && (
        <div
          style={{
            marginTop: '0.75rem',
            padding: '0.65rem 0.85rem',
            background: isDefective ? 'rgba(239, 68, 68, 0.06)' : 'rgba(16, 185, 129, 0.06)',
            borderRadius: 'var(--radius-sm)',
            border: isDefective ? '1px solid rgba(239, 68, 68, 0.25)' : '1px solid rgba(16, 185, 129, 0.25)',
            fontSize: '0.82rem',
          }}
        >
          <span style={{ fontWeight: 650, color: isDefective ? '#ef4444' : '#10b981', display: 'block', marginBottom: '0.25rem' }}>
            Object Classification & Condition Assessment
          </span>
          <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', marginBottom: '0.35rem' }}>
            <span>Recognized Subject: <strong>{detectedObject || 'Object'}</strong></span>
            <span>Condition: <strong style={{ color: isDefective ? '#ef4444' : '#10b981' }}>{conditionLabel}</strong></span>
          </div>
          <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-soft)' }}>
            Dynamic prediction derived from convolutional neural network object classification and surface condition analysis.
          </p>
        </div>
      )}

      <p className="muted" style={{ marginTop: '1rem', fontSize: '0.85rem' }}>
        {result.rgb?.message || result.xray?.message || 'Inspection result returned with no additional message.'}
      </p>

      {localization && (
        <div style={{ marginTop: '1rem', padding: '0.85rem 1rem', background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.4rem', marginBottom: '0.35rem' }}>
            <span style={{ fontWeight: 650, fontSize: '0.82rem' }}>Localization</span>
            <span className="muted" style={{ fontSize: '0.72rem' }}>Statistical anomaly map — not a ground-truth mask</span>
          </div>
          <p className="muted" style={{ margin: 0, fontSize: '0.82rem' }}>
            Method: {localization.method}; anomalous pixels: {localization.anomalous_pixel_count};
            boxes: {localization.bounding_boxes_xywh?.length ?? 0}.
          </p>
        </div>
      )}

      {result.adaptive && (
        <div style={{ marginTop: '0.75rem', padding: '0.85rem 1rem', background: 'var(--lavender)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
          <span style={{ fontWeight: 650, fontSize: '0.82rem', display: 'block', marginBottom: '0.25rem' }}>Adaptive recommendation</span>
          <p className="muted" style={{ margin: 0, fontSize: '0.82rem' }}>
            {result.adaptive.recommendation}: {result.adaptive.reason}
          </p>
        </div>
      )}

      <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', marginTop: '1.25rem' }}>
        <button className="button button-secondary" onClick={() => navigate('/history')}>
          View History
        </button>
        <button
          className="button button-primary"
          onClick={() => navigate(result.id ? `/reports?id=${result.id}` : '/reports', { state: { inspectionId: result.id } })}
        >
          Open Reports <ArrowRight size={14} />
        </button>
      </div>
    </>
  )
}

function ResultMetric({ label, value }) {
  return (
    <div className="result-metric">
      <span className="stat-label">{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
