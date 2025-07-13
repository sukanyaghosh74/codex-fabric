import styles from './FileExplorer.module.css';

const files = [
  {
    name: 'src',
    children: [
      {
        name: 'Controller.js',
        children: [
          { name: 'OrderController.js' },
          { name: 'product.js' },
        ],
      },
      {
        name: 'utils',
        children: [
          { name: 'engtsom' },
          { name: 'utils' },
          { name: 'package' },
        ],
      },
    ],
  },
];

function renderTree(nodes: any[]) {
  return (
    <ul className={styles.tree}>
      {nodes.map((node, i) => (
        <li key={i}>
          <span className={styles.node}>{node.name}</span>
          {node.children && renderTree(node.children)}
        </li>
      ))}
    </ul>
  );
}

export default function FileExplorer() {
  return (
    <div className={styles.panel}>
      <h2>File Explorer</h2>
      <div className={styles.treeWrapper}>{renderTree(files)}</div>
    </div>
  );
} 