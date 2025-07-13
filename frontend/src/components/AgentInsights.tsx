import styles from './AgentInsights.module.css';

export default function AgentInsights() {
  return (
    <div className={styles.panel}>
      <h2>AI Agent Insights</h2>
      <ul className={styles.list}>
        <li>Identified performance bottlenecks in several modules</li>
        <li>Detected dead code in utility functions</li>
        <li>Suggested refactoring of the user authentication logic</li>
      </ul>
      <div className={styles.actions}>
        <button className={styles.button}>Optimize database queries</button>
        <button className={styles.buttonSecondary}>Remove unused functions</button>
      </div>
    </div>
  );
} 