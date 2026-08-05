const fs = require('fs');

try {
  const [graphPath, layersPath, outputPath] = process.argv.slice(2);
  if (!graphPath || !layersPath || !outputPath) throw new Error('Usage: node ua-tour-prepare.js <graph> <layers> <output>');
  const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
  const layers = JSON.parse(fs.readFileSync(layersPath, 'utf8'));
  fs.writeFileSync(outputPath, JSON.stringify({ nodes: graph.nodes || [], edges: graph.edges || [], layers }, null, 2));
} catch (error) {
  console.error(error.stack || error.message);
  process.exit(1);
}
