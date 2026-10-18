import { Component } from 'react'

export default class SceneErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    // Clear, development-time logging so a 3D failure is never silent.
    console.group('AdaptiveInspect: 3D scene render error')
    console.error(error)
    console.error(errorInfo.componentStack)
    console.groupEnd()
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback
    }
    return this.props.children
  }
}
