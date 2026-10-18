import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Float, Points, PointMaterial } from '@react-three/drei'
import { useMemo, useRef, useState, useEffect, useCallback } from 'react'
import * as THREE from 'three'

/* ── Color palette ───────────────────────────────────────────────────── */
// Decorative background (taupe + light blue theme)
const TAUPE         = '#B8A99A'      // warm taupe
const TAUPE_LIGHT   = '#E7DED4'      // light taupe
const ICE_BLUE      = '#DCEEFF'      // ice blue
const PALE_BLUE     = '#EDF5FF'      // pale blue
const NEUTRAL       = '#F8F7F5'      // supporting neutral
const TEXT_DARK     = '#27364B'      // deep blue-grey text

// PCB materials (realistic, not decorative accents)
const SUBSTRATE     = '#0d6b5e'      // deep teal-green PCB substrate
const RESIST_TOP    = '#158a73'      // solder-mask resist surface
const RESIST_EDGE   = '#0a4a42'      // dark edge trim
const COPPER        = '#D4A054'      // warm copper for pads
const TRACE_COLOR   = '#06B6D4'      // cyan traces (engineering detail)
const CHIP_COLOR    = '#1e293b'      // dark slate IC bodies
const CHIP_TOP      = '#334155'      // IC top surface
const CAP_CERAMIC   = '#92400E'      // brown ceramic capacitors
const CAP_ELEC      = '#1e40af'      // blue electrolytic caps
const RESISTOR      = '#1a1a1a'      // dark SMD resistors
const RES_BAND      = '#f59e0b'      // gold resistor band
const PIN_SILVER    = '#c0c0c0'      // silver connector pins
const SOLDER        = '#a8a29e'      // solder point color
const SCAN_COLOR    = '#06B6D4'      // scan beam cyan (required)

/* ── Mouse parallax hook ─────────────────────────────────────────────── */
function usePointer() {
  const pointer = useRef({ x: 0, y: 0 })
  useEffect(() => {
    const handler = (e) => {
      pointer.current.x = (e.clientX / window.innerWidth) - 0.5
      pointer.current.y = (e.clientY / window.innerHeight) - 0.5
    }
    window.addEventListener('mousemove', handler, { passive: true })
    return () => window.removeEventListener('mousemove', handler)
  }, [])
  return pointer
}

/* ── Main PCB board with physical thickness & detailed surface ────── */
function PcbBoard() {
  const groupRef = useRef()
  const pointer = usePointer()

  useFrame((state) => {
    if (!groupRef.current) return
    const t = state.clock.elapsedTime
    const px = pointer.current.x
    const py = pointer.current.y
    groupRef.current.rotation.x = 0.5 + Math.sin(t * 0.15) * 0.03 + py * 0.06
    groupRef.current.rotation.y = -0.3 + Math.cos(t * 0.1) * 0.03 + px * 0.06
  })

  return (
    <group ref={groupRef}>
      {/* Main substrate body */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={[5.0, 0.22, 3.2]} />
        <meshStandardMaterial color={SUBSTRATE} metalness={0.3} roughness={0.8} />
      </mesh>

      {/* Top solder-mask surface */}
      <mesh position={[0, 0.115, 0]} receiveShadow>
        <boxGeometry args={[4.85, 0.02, 3.05]} />
        <meshStandardMaterial color={RESIST_TOP} metalness={0.25} roughness={0.75} />
      </mesh>

      {/* Board edge trim */}
      <mesh position={[0, 0.13, 0]}>
        <boxGeometry args={[5.02, 0.005, 3.22]} />
        <meshStandardMaterial color={RESIST_EDGE} metalness={0.7} roughness={0.3} />
      </mesh>

      {/* Mounting holes */}
      {[[-2.1, -1.3], [2.1, -1.3], [-2.1, 1.3], [2.1, 1.3]].map(([x, z], i) => (
        <group key={`mount-${i}`} position={[x, 0.13, z]}>
          <mesh>
            <cylinderGeometry args={[0.12, 0.12, 0.03, 20]} />
            <meshStandardMaterial color={COPPER} metalness={0.9} roughness={0.15} />
          </mesh>
          <mesh position={[0, 0.005, 0]}>
            <cylinderGeometry args={[0.065, 0.065, 0.04, 20]} />
            <meshStandardMaterial color={RESIST_EDGE} metalness={0.2} roughness={0.9} />
          </mesh>
        </group>
      ))}

      {/* Silkscreen border lines */}
      {[
        { pos: [0, 0.128, -1.45], scale: [4.6, 0.002, 0.012] },
        { pos: [0, 0.128, 1.45], scale: [4.6, 0.002, 0.012] },
        { pos: [-2.3, 0.128, 0], scale: [0.012, 0.002, 2.88] },
        { pos: [2.3, 0.128, 0], scale: [0.012, 0.002, 2.88] },
      ].map((line, i) => (
        <mesh key={`silk-${i}`} position={line.pos}>
          <boxGeometry args={line.scale} />
          <meshStandardMaterial color="#fff" metalness={0} roughness={1} opacity={0.35} transparent />
        </mesh>
      ))}

      <CircuitTraces />
      <IcChips />
      <PassiveComponents />
      <ConnectorPins />
      <SolderPoints />
      <GlowNodes />
      <ScanBeam />
    </group>
  )
}

/* ── Copper traces with realistic routing ────────────────────────────── */
function CircuitTraces() {
  const curves = useMemo(() => [
    // Main data bus
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(-2.2, 0.128, -0.8),
      new THREE.Vector3(-1.2, 0.128, -0.3),
      new THREE.Vector3(-0.1, 0.128, 0.1),
      new THREE.Vector3(0.8, 0.128, -0.2),
      new THREE.Vector3(1.8, 0.128, 0.4),
    ]),
    // Power rail top
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(-2.1, 0.128, -1.2),
      new THREE.Vector3(-0.5, 0.128, -1.15),
      new THREE.Vector3(0.6, 0.128, -1.1),
      new THREE.Vector3(2.0, 0.128, -1.0),
    ]),
    // Power rail bottom
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(-2.1, 0.128, 1.2),
      new THREE.Vector3(-0.8, 0.128, 1.15),
      new THREE.Vector3(0.5, 0.128, 1.0),
      new THREE.Vector3(2.0, 0.128, 0.85),
    ]),
    // Signal route 1
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(-1.8, 0.128, 0.5),
      new THREE.Vector3(-0.8, 0.128, 0.8),
      new THREE.Vector3(0.3, 0.128, 0.5),
      new THREE.Vector3(1.5, 0.128, 0.9),
    ]),
    // Signal route 2
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(-1.5, 0.128, -0.6),
      new THREE.Vector3(-0.3, 0.128, -0.9),
      new THREE.Vector3(0.9, 0.128, -0.5),
      new THREE.Vector3(1.8, 0.128, -0.8),
    ]),
    // Cross connection
    new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.0, 0.128, -1.1),
      new THREE.Vector3(0.15, 0.128, -0.3),
      new THREE.Vector3(0.05, 0.128, 0.5),
      new THREE.Vector3(0.2, 0.128, 1.1),
    ]),
  ], [])

  return (
    <group>
      {curves.map((curve, i) => (
        <mesh key={`trace-${i}`}>
          <tubeGeometry args={[curve, 48, 0.022, 6, false]} />
          <meshStandardMaterial
            color={TRACE_COLOR}
            metalness={0.8}
            roughness={0.3}
            emissive={TRACE_COLOR}
            emissiveIntensity={0.15}
          />
        </mesh>
      ))}

      {/* Contact pads at trace nodes */}
      {[
        [-2.2, -0.8], [-1.2, -0.3], [0.8, -0.2], [1.8, 0.4],
        [-2.1, -1.2], [0.6, -1.1], [2.0, -1.0],
        [-2.1, 1.2], [0.5, 1.0], [2.0, 0.85],
        [-1.8, 0.5], [0.3, 0.5], [1.5, 0.9],
        [-1.5, -0.6], [0.9, -0.5], [1.8, -0.8],
        [0.0, -1.1], [0.2, 1.1],
      ].map(([x, z], i) => (
        <mesh key={`pad-${i}`} position={[x, 0.13, z]} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[0.06, 16]} />
          <meshStandardMaterial color={COPPER} metalness={0.95} roughness={0.12} />
        </mesh>
      ))}
    </group>
  )
}

/* ── IC Chips ─────────────────────────────────────────────────────────── */
function IcChips() {
  const chips = [
    { pos: [-1.3, 0.14, -0.3], size: [0.95, 0.1, 0.55], label: 'Main processor' },
    { pos: [1.0, 0.14, -0.4], size: [0.65, 0.08, 0.4], label: 'Memory' },
    { pos: [-0.4, 0.14, 0.7], size: [0.5, 0.07, 0.35], label: 'Controller' },
    { pos: [1.6, 0.14, 0.6], size: [0.42, 0.06, 0.3], label: 'Regulator' },
  ]

  return (
    <group>
      {chips.map((chip, ci) => (
        <group key={`chip-${ci}`} position={chip.pos}>
          {/* IC body */}
          <mesh castShadow>
            <boxGeometry args={chip.size} />
            <meshStandardMaterial color={CHIP_COLOR} metalness={0.5} roughness={0.6} />
          </mesh>
          {/* IC top surface detail */}
          <mesh position={[0, chip.size[1] / 2 + 0.001, 0]}>
            <boxGeometry args={[chip.size[0] * 0.85, 0.002, chip.size[2] * 0.7]} />
            <meshStandardMaterial color={CHIP_TOP} metalness={0.4} roughness={0.7} />
          </mesh>
          {/* Pin 1 marker dot */}
          <mesh position={[-chip.size[0] * 0.35, chip.size[1] / 2 + 0.002, -chip.size[2] * 0.3]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.02, 12]} />
            <meshStandardMaterial color="#fff" opacity={0.4} transparent />
          </mesh>
          {/* IC legs along sides */}
          {Array.from({ length: Math.ceil(chip.size[0] / 0.12) }, (_, li) => {
            const xOff = -chip.size[0] / 2 + 0.06 + li * 0.12
            return [
              <mesh key={`leg-t-${ci}-${li}`} position={[xOff, chip.size[1] * 0.2, -chip.size[2] / 2 - 0.03]}>
                <boxGeometry args={[0.04, 0.015, 0.06]} />
                <meshStandardMaterial color={PIN_SILVER} metalness={0.9} roughness={0.15} />
              </mesh>,
              <mesh key={`leg-b-${ci}-${li}`} position={[xOff, chip.size[1] * 0.2, chip.size[2] / 2 + 0.03]}>
                <boxGeometry args={[0.04, 0.015, 0.06]} />
                <meshStandardMaterial color={PIN_SILVER} metalness={0.9} roughness={0.15} />
              </mesh>,
            ]
          })}
        </group>
      ))}
    </group>
  )
}

/* ── Passive components: capacitors, resistors ───────────────────────── */
function PassiveComponents() {
  // SMD resistors (dark boxes with gold band)
  const resistors = [
    [0.3, -0.9], [-0.8, -1.0], [1.4, -0.2], [-1.8, 0.9],
    [0.7, 0.3], [-0.2, -0.4], [1.9, -0.6], [-1.2, 0.4],
  ]

  // Ceramic caps (brown cubes)
  const ceramicCaps = [
    [-0.5, -0.7], [0.6, 0.7], [1.3, 0.2], [-1.6, -0.5],
    [0.1, 1.0], [-0.9, 0.2], [1.7, -1.0],
  ]

  // Electrolytic capacitors (taller cylinders)
  const electCaps = [
    [-1.9, -0.6], [2.0, 0.2],
  ]

  return (
    <group>
      {/* SMD Resistors */}
      {resistors.map(([x, z], i) => (
        <group key={`res-${i}`} position={[x, 0.14, z]}>
          <mesh castShadow>
            <boxGeometry args={[0.12, 0.04, 0.06]} />
            <meshStandardMaterial color={RESISTOR} metalness={0.3} roughness={0.8} />
          </mesh>
          {/* Gold band */}
          <mesh position={[0, 0.022, 0]}>
            <boxGeometry args={[0.03, 0.002, 0.062]} />
            <meshStandardMaterial color={RES_BAND} metalness={0.8} roughness={0.2} />
          </mesh>
          {/* End caps */}
          {[-0.055, 0.055].map((xo, ei) => (
            <mesh key={`ec-${ei}`} position={[xo, 0, 0]}>
              <boxGeometry args={[0.015, 0.042, 0.062]} />
              <meshStandardMaterial color={PIN_SILVER} metalness={0.85} roughness={0.2} />
            </mesh>
          ))}
        </group>
      ))}

      {/* Ceramic SMD Capacitors */}
      {ceramicCaps.map(([x, z], i) => (
        <group key={`cc-${i}`} position={[x, 0.14, z]}>
          <mesh castShadow>
            <boxGeometry args={[0.1, 0.05, 0.07]} />
            <meshStandardMaterial color={CAP_CERAMIC} metalness={0.25} roughness={0.7} />
          </mesh>
          {[-0.045, 0.045].map((xo, ei) => (
            <mesh key={`cce-${ei}`} position={[xo, 0, 0]}>
              <boxGeometry args={[0.015, 0.052, 0.072]} />
              <meshStandardMaterial color={PIN_SILVER} metalness={0.85} roughness={0.2} />
            </mesh>
          ))}
        </group>
      ))}

      {/* Electrolytic Capacitors */}
      {electCaps.map(([x, z], i) => (
        <group key={`elec-${i}`} position={[x, 0.2, z]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.12, 0.12, 0.2, 16]} />
            <meshStandardMaterial color={CAP_ELEC} metalness={0.4} roughness={0.5} />
          </mesh>
          {/* Top vent pattern */}
          <mesh position={[0, 0.101, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.1, 16]} />
            <meshStandardMaterial color="#1e3a8a" metalness={0.3} roughness={0.6} />
          </mesh>
          {/* Polarity stripe */}
          <mesh position={[0.11, 0, 0]}>
            <boxGeometry args={[0.02, 0.19, 0.06]} />
            <meshStandardMaterial color="#93c5fd" metalness={0.1} roughness={0.9} />
          </mesh>
        </group>
      ))}
    </group>
  )
}

/* ── Connector pins ──────────────────────────────────────────────────── */
function ConnectorPins() {
  return (
    <group>
      {/* Header connector */}
      <group position={[2.15, 0.17, 0]}>
        <mesh castShadow>
          <boxGeometry args={[0.2, 0.12, 1.4]} />
          <meshStandardMaterial color={CHIP_COLOR} metalness={0.4} roughness={0.7} />
        </mesh>
        {Array.from({ length: 10 }, (_, i) => (
          <mesh key={`hpin-${i}`} position={[0, 0.1, -0.63 + i * 0.14]}>
            <boxGeometry args={[0.02, 0.12, 0.02]} />
            <meshStandardMaterial color={COPPER} metalness={0.95} roughness={0.1} />
          </mesh>
        ))}
      </group>

      {/* Small 4-pin connector */}
      <group position={[-2.1, 0.15, 0.8]}>
        <mesh castShadow>
          <boxGeometry args={[0.15, 0.1, 0.5]} />
          <meshStandardMaterial color="#fff" metalness={0.1} roughness={0.9} opacity={0.85} transparent />
        </mesh>
        {Array.from({ length: 4 }, (_, i) => (
          <mesh key={`spin-${i}`} position={[0, 0.08, -0.18 + i * 0.12]}>
            <boxGeometry args={[0.015, 0.08, 0.015]} />
            <meshStandardMaterial color={COPPER} metalness={0.95} roughness={0.1} />
          </mesh>
        ))}
      </group>
    </group>
  )
}

/* ── Solder points ───────────────────────────────────────────────────── */
function SolderPoints() {
  const points = useMemo(() => [
    [-1.7, 0.7], [-1.3, 0.8], [-0.9, 0.9],
    [0.8, -0.8], [1.1, -0.9], [1.4, -0.7],
    [-0.5, 0.3], [0.4, -0.1], [1.6, 0.3],
  ], [])

  return (
    <group>
      {points.map(([x, z], i) => (
        <mesh key={`solder-${i}`} position={[x, 0.13, z]} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[0.035, 12]} />
          <meshStandardMaterial color={SOLDER} metalness={0.7} roughness={0.25} />
        </mesh>
      ))}
    </group>
  )
}

/* ── Glowing circuit nodes ───────────────────────────────────────────── */
function GlowNodes() {
  const ref = useRef()

  useFrame((state) => {
    if (!ref.current) return
    const mat = ref.current.material
    if (mat) {
      mat.opacity = 0.5 + Math.sin(state.clock.elapsedTime * 2) * 0.3
    }
  })

  const positions = useMemo(() => {
    const pts = [
      [-2.2, -0.8], [-1.2, -0.3], [0.8, -0.2], [1.8, 0.4],
      [-2.1, -1.2], [2.0, -1.0], [-2.1, 1.2], [2.0, 0.85],
      [-1.8, 0.5], [1.5, 0.9], [0.0, -1.1], [0.2, 1.1],
      [-0.8, -1.0], [1.4, -0.2], [0.7, 0.3], [-1.2, 0.4],
      [0.6, 0.7], [-0.9, 0.2], [-0.5, -0.7], [1.3, 0.2],
    ]
    const arr = new Float32Array(pts.length * 3)
    pts.forEach(([x, z], i) => {
      arr[i * 3] = x
      arr[i * 3 + 1] = 0.14
      arr[i * 3 + 2] = z
    })
    return arr
  }, [])

  return (
    <Points ref={ref} positions={positions} stride={3}>
      <PointMaterial
        transparent
        color={NODE_ICE}
        size={0.06}
        sizeAttenuation
        opacity={0.8}
        depthWrite={false}
      />
    </Points>
  )
}

/* ── Translucent cyan scan beam ──────────────────────────────────────── */
function ScanBeam() {
  const ref = useRef()

  useFrame((state) => {
    if (!ref.current) return
    const t = state.clock.elapsedTime
    ref.current.position.x = Math.sin(t * 0.5) * 2.2
    ref.current.material.opacity = 0.25 + Math.sin(t * 1.5) * 0.08
  })

  return (
    <mesh ref={ref} position={[0, 0.16, 0]}>
      <boxGeometry args={[0.04, 0.08, 3.0]} />
      <meshStandardMaterial
        color={SCAN_COLOR}
        transparent
        opacity={0.3}
        emissive={SCAN_COLOR}
        emissiveIntensity={2.5}
        metalness={0.5}
        roughness={0.2}
        depthWrite={false}
      />
    </mesh>
  )
}

/* ── Ambient floating particles ──────────────────────────────────────── */
function ParticleField() {
  const ref = useRef()
  const positions = useMemo(() => {
    const count = 120
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      arr[i * 3]     = (Math.random() - 0.5) * 12
      arr[i * 3 + 1] = (Math.random() - 0.5) * 6
      arr[i * 3 + 2] = (Math.random() - 0.5) * 10
    }
    return arr
  }, [])

  useFrame((state, delta) => {
    if (!ref.current) return
    ref.current.rotation.y += delta * 0.012
    ref.current.position.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.08
  })

  return (
    <Points ref={ref} positions={positions} stride={3}>
      <PointMaterial
        transparent
        color={NODE_LAV}
        size={0.018}
        sizeAttenuation
        depthWrite={false}
        opacity={0.45}
      />
    </Points>
  )
}

/* ── Decorative background: geometric forms, ribbons, floating elements ─ */
function DecorativeBackground() {
  const groupRef = useRef()
  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.006
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.15) * 0.05
    }
  })

  return (
    <group ref={groupRef}>
      {/* Large floating torus ribbon (taupe) */}
      <mesh position={[-3.5, 0.5, -4]} rotation={[0.3, 0.5, 0.2]}>
        <torusGeometry args={[1.8, 0.08, 16, 64]} />
        <meshStandardMaterial color={TAUPE_LIGHT} metalness={0.3} roughness={0.7} transparent opacity={0.35} />
      </mesh>

      {/* Flowing ribbon (ice blue) */}
      <mesh position={[-2.0, -0.5, -3]} rotation={[0.2, 0.3, 0.1]}>
        <torusGeometry args={[1.4, 0.05, 12, 48]} />
        <meshStandardMaterial color={ICE_BLUE} metalness={0.2} roughness={0.8} transparent opacity={0.25} />
      </mesh>

      {/* Large sphere (lavender) */}
      <mesh position={[-4.0, 1.2, -5]}>
        <sphereGeometry args={[0.9, 24, 24]} />
        <meshStandardMaterial color={TAUPE_LIGHT} metalness={0.15} roughness={0.9} transparent opacity={0.2} />
      </mesh>

      {/* Floating box (pale blue) */}
      <mesh position={[-2.5, -1.0, -2]} rotation={[0.4, 0.6, 0.3]}>
        <boxGeometry args={[0.6, 0.6, 0.6]} />
        <meshStandardMaterial color={PALE_BLUE} metalness={0.2} roughness={0.85} transparent opacity={0.2} />
      </mesh>

      {/* Floating octahedron (taupe) */}
      <mesh position={[-4.5, -0.2, -3.5]} rotation={[0.2, 0.4, 0.1]}>
        <octahedronGeometry args={[0.5, 0]} />
        <meshStandardMaterial color={TAUPE} metalness={0.3} roughness={0.7} transparent opacity={0.25} />
      </mesh>

      {/* Fine circuit lines (background detail) */}
      <mesh position={[-3.0, 0.0, -2.5]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[6, 4]} />
        <meshStandardMaterial color={ICE_BLUE} transparent opacity={0.06} metalness={0.1} roughness={0.95} />
      </mesh>

      {/* Floating particles (taupe/blue) */}
      <Points positions={(() => {
        const arr = new Float32Array(80 * 3)
        for (let i = 0; i < 80; i++) {
          arr[i * 3] = (Math.random() - 0.5) * 14
          arr[i * 3 + 1] = (Math.random() - 0.5) * 7
          arr[i * 3 + 2] = (Math.random() - 0.5) * 10
        }
        return arr
      })()} stride={3}>
        <PointMaterial transparent color={TAUPE} size={0.025} sizeAttenuation depthWrite={false} opacity={0.35} />
      </Points>
    </group>
  )
}

/* ── Camera rig: frames the PCB on the right + decorative background ── */
function CameraRig() {
  const { camera } = useThree()
  useFrame(() => {
    camera.lookAt(2.0, 0, 0)
  })
  return null
}

/* ── Complete scene content ──────────────────────────────────────────── */
function SceneContent() {
  return (
    <>
      <CameraRig />
      {/* Lighting */}
      <ambientLight intensity={0.7} />
      <directionalLight
        position={[6, 8, 5]}
        intensity={1.1}
        color="#ffffff"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-near={0.5}
        shadow-camera-far={50}
        shadow-camera-left={-5}
        shadow-camera-right={5}
        shadow-camera-top={5}
        shadow-camera-bottom={-5}
      />
      <directionalLight position={[-4, 3, -6]} intensity={0.3} color="#e0f2fe" />
      <pointLight position={[-4, 4, 3]} intensity={0.5} color="#06B6D4" distance={12} />
      <pointLight position={[4, 2, -3]} intensity={0.35} color="#818cf8" distance={10} />

      {/* Decorative background (left + center) */}
      <DecorativeBackground />

      {/* PCB on the RIGHT side */}
      <group position={[3.0, 0, 0]}>
        <Float speed={0.8} rotationIntensity={0.02} floatIntensity={0.15}>
          <PcbBoard />
        </Float>
      </group>

      <ParticleField />
    </>
  )
}

/* ── Exported component ──────────────────────────────────────────────── */
export default function HeroScene() {
  return (
    <Canvas
      className="hero-canvas"
      camera={{ position: [3.0, 3.5, 7.5], fov: 38, near: 0.1, far: 100 }}
      dpr={[1, 1.5]}
      gl={{ antialias: true, alpha: true, preserveDrawingBuffer: false, powerPreference: 'default' }}
      shadows
      style={{ background: 'transparent' }}
    >
      <SceneContent />
    </Canvas>
  )
}
