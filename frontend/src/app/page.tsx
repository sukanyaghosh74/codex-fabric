import styles from "./page.module.css";
import KnowledgeGraph from "../components/KnowledgeGraph";
import RecentAIActions from "../components/RecentAIActions";
import AgentInsights from "../components/AgentInsights";
import FileExplorer from "../components/FileExplorer";
import CodeHealthDashboard from "../components/CodeHealthDashboard";
import AgentChat from "../components/AgentChat";

export default function Dashboard() {
  return (
    <div className={styles.dashboard}>
      <header className={styles.header}>
        <h1>CODEX FABRIC</h1>
        <div className={styles.headerIcons}>
          <span className={styles.icon} />
          <span className={styles.icon} />
        </div>
      </header>
      <main className={styles.grid}>
        <section className={styles.knowledgeGraph}><KnowledgeGraph /></section>
        <section className={styles.recentAIActions}><RecentAIActions /></section>
        <section className={styles.agentInsights}><AgentInsights /></section>
        <section className={styles.fileExplorer}><FileExplorer /></section>
        <section className={styles.codeHealth}><CodeHealthDashboard /></section>
        <section className={styles.agentChat}><AgentChat /></section>
      </main>
      <footer className={styles.footer}>
        Made with <span className={styles.heart}>💗</span> by Sukanya Ghosh
      </footer>
    </div>
  );
}
