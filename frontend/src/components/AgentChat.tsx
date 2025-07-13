"use client";

import styles from './AgentChat.module.css';

const messages = [
  { sender: 'Senior DevBot', text: "What's the most pritle part of this system?" },
  { sender: 'You', text: 'Refactor the login flow for extensibility' },
];

export default function AgentChat() {
  return (
    <div className={styles.panel}>
      <h2>Agent Chat</h2>
      <div className={styles.chatHistory}>
        {messages.map((msg, i) => (
          <div key={i} className={msg.sender === 'You' ? styles.userMsg : styles.botMsg}>
            <span className={styles.sender}>{msg.sender}:</span> {msg.text}
          </div>
        ))}
      </div>
      <form className={styles.inputRow} onSubmit={e => e.preventDefault()}>
        <input className={styles.input} placeholder="Type your message..." />
        <button className={styles.sendBtn} type="submit">Send</button>
      </form>
    </div>
  );
} 