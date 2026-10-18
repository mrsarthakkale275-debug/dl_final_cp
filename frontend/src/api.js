export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api'

async function request(path, options = {}) {
  const url = `${API_BASE_URL}${path}`
  let response
  try {
    response = await fetch(url, options)
  } catch (error) {
    throw new Error('The backend is not reachable. Start the FastAPI server and reload.')
  }

  const contentType = response.headers.get('content-type') || ''
  let body = null
  if (contentType.includes('application/json')) {
    body = await response.json()
  } else {
    body = await response.text()
  }

  if (!response.ok) {
    const message =
      body && typeof body === 'object' && body.error && body.error.message
        ? body.error.message
        : typeof body === 'string'
          ? body
          : `Request failed with status ${response.status}`
    throw new Error(message)
  }

  return body
}

export async function getHealth() {
  return request('/health')
}

export async function getModelStatus() {
  return request('/model-status')
}

export async function getDatasetStatus() {
  return request('/datasets/status')
}

export async function getDashboardSummary() {
  return request('/dashboard/summary')
}

export async function inspectImage(endpoint, file, options = {}) {
  const formData = new FormData()
  formData.append('file', file)
  if (options.xrayFile) {
    formData.append('xray_file', options.xrayFile)
  }

  return request(`/inspect/${endpoint}`, {
    method: 'POST',
    body: formData,
  })
}

export async function listInspections(params = {}) {
  const query = new URLSearchParams(params).toString()
  return request(`/inspections${query ? `?${query}` : ''}`)
}

export async function getInspection(id) {
  return request(`/inspections/${id}`)
}

export async function deleteInspection(id) {
  return request(`/inspections/${id}`, { method: 'DELETE' })
}

export async function getReport(id, format = 'json') {
  return request(`/inspections/${id}/report?format=${format}`, {
    headers: { accept: format === 'markdown' ? 'text/markdown' : 'application/json' },
  })
}