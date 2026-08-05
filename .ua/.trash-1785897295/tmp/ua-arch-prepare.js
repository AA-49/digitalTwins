const fs = require('fs');
const [graphPath, outputPath] = process.argv.slice(2);
if (!graphPath || !outputPath) process.exit(1);
const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
const fileTypes = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
const eligible = graph.nodes.filter((node) => fileTypes.has(node.type));
const byPath = new Map();
for (const node of eligible) {
  const current = byPath.get(node.filePath);
  const canonical = node.id === `${node.type}:${node.filePath}`;
  if (!current || canonical) byPath.set(node.filePath, node);
}
const fileNodes = [...byPath.values()].map(({ id, type, name, filePath, summary, tags }) => ({ id, type, name, filePath, summary, tags }));
const ids = new Set(fileNodes.map((node) => node.id));
const allEdges = graph.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target));
const importEdges = allEdges.filter((edge) => edge.type === 'imports');
fs.writeFileSync(outputPath, JSON.stringify({ fileNodes, importEdges, allEdges }, null, 2) + '\n');
