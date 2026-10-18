import { NavLink, Outlet, useLocation, useNavigate, Link } from 'react-router-dom'
import { Activity, BarChart3, FileText, History, PlusCircle, Info, Menu, X } from 'lucide-react'
import { useState } from 'react'

const navigation = [
  { to: '/overview',     icon: BarChart3,   label: 'Overview' },
  { to: '/inspection',  icon: PlusCircle,  label: 'Inspection' },
  { to: '/history',     icon: History,     label: 'History' },
  { to: '/reports',     icon: FileText,    label: 'Reports' },
  { to: '/model-status', icon: Activity,   label: 'Model Status' },
  { to: '/about',       icon: Info,        label: 'About' },
]

export default function AppShell() {
  const location = useLocation()
  const navigate = useNavigate()
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="nav-bar">
          <Link to="/" className="brand" aria-label="AdaptiveInspect AI Home">
            <div className="brand-mark">AI</div>
            <div>
              <div className="brand-name">AdaptiveInspect AI</div>
              <div className="brand-subtitle">PCB quality inspection</div>
            </div>
          </Link>

          <nav className="nav-links" aria-label="Primary navigation">
            {navigation.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
              >
                <Icon size={15} />
                {label}
              </NavLink>
            ))}
          </nav>

          <div className="nav-actions">
            <button
              className="button button-primary button-sm"
              onClick={() => navigate('/inspection')}
              id="nav-new-inspection"
            >
              <PlusCircle size={14} />
              New Inspection
            </button>
          </div>
        </div>
      </header>

      <main className="app-main">
        <Outlet />
      </main>
    </div>
  )
}
