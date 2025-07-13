import styles from './KnowledgeGraph.module.css';

export default function KnowledgeGraph() {
  return (
    <div className={styles.panel}>
      <h2>Codebase Knowledge Graph</h2>
      <div className={styles.graphArea}>
        {/* Larger, denser SVG web */}
        <svg viewBox="0 0 600 350" className={styles.graphSvg}>
          {/* Edges */}
          <line x1="100" y1="80" x2="200" y2="60" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="200" y1="60" x2="300" y2="100" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="300" y1="100" x2="400" y2="60" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="400" y1="60" x2="500" y2="120" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="100" y1="80" x2="150" y2="200" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="150" y1="200" x2="250" y2="250" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="250" y1="250" x2="350" y2="220" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="350" y1="220" x2="450" y2="300" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="500" y1="120" x2="450" y2="300" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="300" y1="100" x2="350" y2="220" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="200" y1="60" x2="250" y2="250" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="400" y1="60" x2="350" y2="220" stroke="#3a8b8b" strokeWidth="2" />
          <line x1="150" y1="200" x2="300" y2="100" stroke="#3a8b8b" strokeWidth="2" />
          {/* Nodes */}
          <circle cx="100" cy="80" r="12" fill="#4de0e0" />
          <circle cx="200" cy="60" r="12" fill="#4de0e0" />
          <circle cx="300" cy="100" r="14" fill="#4de0e0" />
          <circle cx="400" cy="60" r="12" fill="#4de0e0" />
          <circle cx="500" cy="120" r="12" fill="#4de0e0" />
          <circle cx="150" cy="200" r="12" fill="#4de0e0" />
          <circle cx="250" cy="250" r="12" fill="#4de0e0" />
          <circle cx="350" cy="220" r="12" fill="#4de0e0" />
          <circle cx="450" cy="300" r="12" fill="#4de0e0" />
        </svg>
        {/* Mock tooltip on main node */}
        <div className={styles.tooltip} style={{ left: '290px', top: '60px' }}>
          <strong>OrderController</strong><br />
          Type: Controller<br />
          Coupling: 16<br />
          Degree: 18<br />
          Refactor Score: 7
        </div>
      </div>
    </div>
  );
} 