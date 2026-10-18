import { Navigate, Route, Routes, useLocation } from 'react-router-dom'

import AppShell from './components/AppShell.jsx'
import Landing from './pages/Landing.jsx'
import Dashboard from './pages/Dashboard.jsx'
import InspectionWorkspace from './pages/InspectionWorkspace.jsx'
import History from './pages/History.jsx'
import Reports from './pages/Reports.jsx'
import ModelStatus from './pages/ModelStatus.jsx'
import About from './pages/About.jsx'

export default function App() {
  const location = useLocation()
  const landing = location.pathname === '/'

  return (
    <div className="app-root">
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route element={<AppShell />}>
          <Route path="/overview" element={<Dashboard />} />
          <Route path="/inspection" element={<InspectionWorkspace />} />
          <Route path="/history" element={<History />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/model-status" element={<ModelStatus />} />
          <Route path="/about" element={<About />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      {!landing && null}
    </div>
  )
}
