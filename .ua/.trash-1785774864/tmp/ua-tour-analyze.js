const fs = require('fs');

function basename(path) {
  return String(path || '').replace(/\\/g, '/').split('/').pop();
}

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) throw new Error('Usage: node ua-tour-analyze.js <input> <output>');
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = input.nodes || [];
  const edges = input.edges || [];
  const layers = input.layers || [];
  const byId = new Map(nodes.map(node => [node.id, node]));
  const fanIn = new Map(nodes.map(node => [node.id, 0]));
  const fanOut = new Map(nodes.map(node => [node.id, 0]));
  for (const edge of edges) {
    if (fanOut.has(edge.source)) fanOut.set(edge.source, fanOut.get(edge.source) + 1);
    if (fanIn.has(edge.target)) fanIn.set(edge.target, fanIn.get(edge.target) + 1);
  }
  const rank = (counts, key) => nodes.map(node => ({ id: node.id, [key]: counts.get(node.id), name: node.name }))
    .sort((a, b) => b[key] - a[key] || a.id.localeCompare(b.id)).slice(0, 20);
  const fanInRanking = rank(fanIn, 'fanIn');
  const fanOutRanking = rank(fanOut, 'fanOut');
  const outValues = [...fanOut.values()].sort((a, b) => b - a);
  const inValues = [...fanIn.values()].sort((a, b) => a - b);
  const topOutThreshold = outValues[Math.max(0, Math.ceil(outValues.length * 0.10) - 1)] || 0;
  const bottomInThreshold = inValues[Math.max(0, Math.ceil(inValues.length * 0.25) - 1)] || 0;
  const entryNames = new Set(['index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js','mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py','Application.java','Main.java','Program.cs','config.ru','index.php','App.swift','Application.kt','main.cpp','main.c']);
  const candidates = nodes.map(node => {
    const path = String(node.filePath || node.name || '').replace(/\\/g, '/');
    const name = basename(path);
    const depth = path.split('/').filter(Boolean).length;
    let score = 0;
    if (node.type === 'file') {
      if (entryNames.has(name)) score += 3;
      if (depth <= 2) score += 1;
      if ((fanOut.get(node.id) || 0) >= topOutThreshold) score += 1;
      if ((fanIn.get(node.id) || 0) <= bottomInThreshold) score += 1;
    } else if (node.type === 'document') {
      if (path === 'README.md') score += 5;
      else if (depth === 1 && name.endsWith('.md')) score += 2;
    }
    return { id: node.id, score, name: node.name, summary: node.summary || '' };
  }).filter(item => item.score > 0).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, 5);
  const codeStart = candidates.map(item => byId.get(item.id)).find(node => node && node.type === 'file');
  const allowed = new Set(['imports', 'calls']);
  const adjacency = new Map(nodes.map(node => [node.id, []]));
  for (const edge of edges) if (allowed.has(edge.type) && adjacency.has(edge.source) && byId.has(edge.target)) adjacency.get(edge.source).push(edge.target);
  for (const targets of adjacency.values()) targets.sort();
  const order = [], depthMap = {}, byDepth = {};
  if (codeStart) {
    const queue = [codeStart.id];
    depthMap[codeStart.id] = 0;
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      const depth = depthMap[id];
      (byDepth[depth] ||= []).push(id);
      for (const target of adjacency.get(id) || []) if (!(target in depthMap)) {
        depthMap[target] = depth + 1;
        queue.push(target);
      }
    }
  }
  const info = node => ({ id: node.id, name: node.name, type: node.type, summary: node.summary || '' });
  const nonCodeFiles = {
    documentation: nodes.filter(n => n.type === 'document').map(info),
    infrastructure: nodes.filter(n => ['service','pipeline','resource'].includes(n.type)).map(info),
    data: nodes.filter(n => ['table','schema','endpoint'].includes(n.type)).map(info),
    config: nodes.filter(n => n.type === 'config').map(info)
  };
  const relation = new Set(edges.filter(e => ['imports','calls'].includes(e.type)).map(e => `${e.source}\u0000${e.target}\u0000${e.type}`));
  const seedPairs = [];
  for (const edge of edges) if (['imports','calls'].includes(edge.type) && relation.has(`${edge.target}\u0000${edge.source}\u0000${edge.type}`) && edge.source < edge.target) seedPairs.push([edge.source, edge.target]);
  const clusters = [];
  for (const pair of seedPairs) {
    const cluster = new Set(pair);
    let changed = true;
    while (changed && cluster.size < 5) {
      changed = false;
      for (const node of nodes) {
        if (cluster.has(node.id)) continue;
        let links = 0;
        for (const member of cluster) if (edges.some(e => (e.source === node.id && e.target === member) || (e.source === member && e.target === node.id))) links++;
        if (links >= 2) { cluster.add(node.id); changed = true; if (cluster.size >= 5) break; }
      }
    }
    const ids = [...cluster].sort();
    const signature = ids.join('|');
    if (!clusters.some(c => c.signature === signature)) {
      const edgeCount = edges.filter(e => cluster.has(e.source) && cluster.has(e.target)).length;
      clusters.push({ nodes: ids, edgeCount, signature });
    }
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount || a.signature.localeCompare(b.signature));
  const nodeSummaryIndex = Object.fromEntries(nodes.map(node => [node.id, { name: node.name, type: node.type, summary: node.summary || '' }]));
  const result = {
    scriptCompleted: true,
    entryPointCandidates: candidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: { startNode: codeStart ? codeStart.id : null, order, depthMap, byDepth },
    nonCodeFiles,
    clusters: clusters.slice(0, 10).map(({ nodes, edgeCount }) => ({ nodes, edgeCount })),
    layers: { count: layers.length, list: layers.map(({ id, name, description }) => ({ id, name, description })) },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length
  };
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
} catch (error) {
  console.error(error.stack || error.message);
  process.exit(1);
}
