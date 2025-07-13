import styles from './RecentAIActions.module.css';

const actions = [
  { action: 'Refactor duplicated code in PaymentService', time: '7m' },
  { action: 'Optimize loop in shopping_cart.py', time: '3m' },
  { action: 'Add debug scc', time: '4s' },
];

export default function RecentAIActions() {
  return (
    <div className={styles.panel}>
      <h2>Recent AI Actions</h2>
      <ul className={styles.list}>
        {actions.map((a, i) => (
          <li key={i} className={styles.item}>
            <span>{a.action}</span>
            <span className={styles.time}>{a.time}</span>
          </li>
        ))}
      </ul>
    </div>
  );
} 