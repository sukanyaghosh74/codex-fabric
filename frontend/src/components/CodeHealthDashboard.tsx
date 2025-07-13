import styles from './CodeHealthDashboard.module.css';

export default function CodeHealthDashboard() {
  return (
    <div className={styles.panel}>
      <h2>Code Health Dashboard</h2>
      <div className={styles.metricRow}>
        <span>Cyclomatic Complexity</span>
        <div className={styles.barWrapper}><div className={styles.bar} style={{width: '41%'}} /></div>
        <span className={styles.value}>41</span>
      </div>
      <div className={styles.metricRow}>
        <span>Duplication</span>
        <div className={styles.barWrapper}><div className={styles.bar} style={{width: '7%'}} /></div>
        <span className={styles.value}>7%</span>
      </div>
      <div className={styles.metricRow}>
        <span>Test Coverage</span>
        <div className={styles.barWrapper}><div className={styles.bar} style={{width: '85%'}} /></div>
        <span className={styles.value}>85%</span>
      </div>
      <div className={styles.metricRow}>
        <span>Maintainability Index</span>
        <div className={styles.barWrapper}><div className={styles.bar} style={{width: '16%'}} /></div>
        <span className={styles.value}>16</span>
      </div>
    </div>
  );
} 