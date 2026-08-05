const fs = require('fs');
const path = require('path');

function fatal(message) {
  process.stderr.write(String(message) + '\n');
  process.exit(1);
}

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) fatal('Usage: node ua-arch-analyze.js INPUT OUTPUT');

try {
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const fileNodes = input.fileNodes || [];
  const importEdges = input.importEdges || [];
  const nodeIds = new Set(fileNodes.map((node) => node.id));
  const allEdges = (input.allEdges || []).filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
  const byId = new Map(fileNodes.map((node) => [node.id, node]));
  const clean = (p) => String(p || '').replace(/\\/g, '/').replace(/^\.\//, '');
  const paths = fileNodes.map((node) => clean(node.filePath));

  function commonDirectoryPrefix(values) {
    if (!values.length) return [];
    const segments = values.map((value) => value.split('/').slice(0, -1));
    const prefix = [];
    for (let i = 0; i < Math.min(...segments.map((parts) => parts.length)); i++) {
      if (segments.every((parts) => parts[i] === segments[0][i])) prefix.push(segments[0][i]);
      else break;
    }
    return prefix;
  }

  const prefix = commonDirectoryPrefix(paths);
  const topGroups = paths.map((p) => {
    const parts = p.split('/');
    const rest = parts.slice(prefix.length);
    return rest.length > 1 ? rest[0] : 'root';
  });
  const isFlat = new Set(topGroups).size === 1 && topGroups[0] === 'root';
  function flatGroup(node) {
    const p = clean(node.filePath).toLowerCase();
    const base = path.posix.basename(p);
    if (/^(test_.*\.py|.*\.(test|spec)\.)/.test(base)) return 'test';
    if (/dockerfile|docker-compose/.test(base)) return 'infrastructure';
    if (/\.(md|rst)$/.test(base)) return 'documentation';
    if (/\.(ya?ml|json|toml|ini|env)$/.test(base) || base.startsWith('.env')) return 'config';
    return path.posix.extname(base).replace('.', '') || 'other';
  }
  function groupFor(node) {
    if (isFlat) return flatGroup(node);
    const parts = clean(node.filePath).split('/').slice(prefix.length);
    return parts.length > 1 ? parts[0] : 'root';
  }

  const directoryGroups = {};
  const nodeTypeGroups = {};
  const groupById = new Map();
  for (const node of fileNodes) {
    const group = groupFor(node);
    groupById.set(node.id, group);
    (directoryGroups[group] ||= []).push(node.id);
    (nodeTypeGroups[node.type] ||= []).push(node.id);
  }

  const fileFanIn = Object.fromEntries(fileNodes.map((node) => [node.id, 0]));
  const fileFanOut = Object.fromEntries(fileNodes.map((node) => [node.id, 0]));
  const groupImportsFrom = Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, []]));
  const groupImportedBy = Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, []]));
  const adjacencySets = Object.fromEntries(fileNodes.map((node) => [node.id, new Set()]));
  const interCounts = new Map();
  for (const edge of importEdges) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue;
    adjacencySets[edge.source].add(edge.target);
    const from = groupById.get(edge.source);
    const to = groupById.get(edge.target);
    if (from !== to) interCounts.set(`${from}\u0000${to}`, (interCounts.get(`${from}\u0000${to}`) || 0) + 1);
  }
  for (const [source, targets] of Object.entries(adjacencySets)) {
    fileFanOut[source] = targets.size;
    for (const target of targets) fileFanIn[target]++;
  }
  for (const key of interCounts.keys()) {
    const [from, to] = key.split('\u0000');
    groupImportsFrom[from].push(to);
    groupImportedBy[to].push(from);
  }
  for (const obj of [groupImportsFrom, groupImportedBy]) for (const key of Object.keys(obj)) obj[key] = [...new Set(obj[key])].sort();
  const interGroupImports = [...interCounts.entries()].map(([key, count]) => {
    const [from, to] = key.split('\u0000');
    return { from, to, count };
  }).sort((a, b) => a.from.localeCompare(b.from) || a.to.localeCompare(b.to));

  const intraGroupDensity = {};
  for (const group of Object.keys(directoryGroups)) {
    let internalEdges = 0;
    let totalEdges = 0;
    for (const edge of importEdges) {
      if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue;
      const from = groupById.get(edge.source);
      const to = groupById.get(edge.target);
      if (from === group || to === group) totalEdges++;
      if (from === group && to === group) internalEdges++;
    }
    intraGroupDensity[group] = { internalEdges, totalEdges, density: totalEdges ? internalEdges / totalEdges : 0 };
  }

  const patternRules = [
    [/^(routes|api|controllers|endpoints|handlers|blueprints|serializers)$/i, 'api'],
    [/^(services|core|lib|domain|logic|internal|composables|mailers|jobs|channels|signals)$/i, 'service'],
    [/^(models|db|data|persistence|repository|entities|migrations|sql|database|schema|entity)$/i, 'data'],
    [/^(components|views|pages|ui|layouts|screens)$/i, 'ui'],
    [/^(middleware|plugins|interceptors|guards)$/i, 'middleware'],
    [/^(utils|helpers|common|shared|tools|pkg|templatetags)$/i, 'utility'],
    [/^(config|constants|env|settings|management|commands)$/i, 'config'],
    [/^(__tests__|test|tests|spec|specs|src\/test\/java)$/i, 'test'],
    [/^(types|interfaces|schemas|contracts|dtos|dto|request|response)$/i, 'types'],
    [/^(hooks)$/i, 'hooks'], [/^(store|state|reducers|actions|slices)$/i, 'state'],
    [/^(assets|static|public|templates)$/i, 'assets'], [/^(cmd|bin)$/i, 'entry'],
    [/^(docs|documentation|wiki)$/i, 'documentation'],
    [/^(deploy|deployment|infra|infrastructure|docker|k8s|kubernetes|helm|charts|terraform|tf)$/i, 'infrastructure'],
    [/^(\.github|\.gitlab|\.circleci)$/i, 'ci-cd']
  ];
  const patternMatches = {};
  for (const group of Object.keys(directoryGroups)) {
    const match = patternRules.find(([regex]) => regex.test(group));
    if (match) patternMatches[group] = match[1];
  }

  const crossMap = new Map();
  for (const edge of allEdges) {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (!source || !target || source.type === target.type) continue;
    const key = `${source.type}\u0000${target.type}\u0000${edge.type}`;
    crossMap.set(key, (crossMap.get(key) || 0) + 1);
  }
  const crossCategoryEdges = [...crossMap.entries()].map(([key, count]) => {
    const [fromType, toType, edgeType] = key.split('\u0000');
    return { fromType, toType, edgeType, count };
  });
  const nonCodeConnections = allEdges.filter((edge) => {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    return source && target && source.type !== 'file' && target.type === 'file';
  });

  const lowerPaths = paths.map((p) => p.toLowerCase());
  const infraFiles = fileNodes.filter((node) => {
    const p = clean(node.filePath).toLowerCase();
    return node.type === 'service' || node.type === 'pipeline' || /(^|\/)(dockerfile[^/]*|docker-compose[^/]*\.ya?ml|.*\.tf(vars)?|jenkinsfile|\.gitlab-ci\.ya?ml)$/.test(p) || /(^|\/)(k8s|kubernetes|helm|charts|terraform|\.github\/workflows)(\/|$)/.test(p);
  }).map((node) => node.filePath);
  const deploymentTopology = {
    hasDockerfile: lowerPaths.some((p) => /(^|\/)dockerfile[^/]*$/.test(p)),
    hasCompose: lowerPaths.some((p) => /(^|\/)docker-compose[^/]*\.ya?ml$/.test(p)),
    hasK8s: lowerPaths.some((p) => /(^|\/)(k8s|kubernetes|helm|charts)(\/|$)/.test(p)),
    hasTerraform: lowerPaths.some((p) => /\.tf(vars)?$/.test(p) || /(^|\/)terraform\//.test(p)),
    hasCI: lowerPaths.some((p) => /(^|\/)\.github\/workflows\//.test(p) || /(^|\/)(\.gitlab-ci\.ya?ml|jenkinsfile)$/.test(p)),
    infraFiles: [...new Set(infraFiles)]
  };

  const dataPipeline = {
    schemaFiles: fileNodes.filter((n) => n.type === 'schema' || /\.(sql|graphql|gql|proto)$/.test(clean(n.filePath).toLowerCase())).map((n) => n.filePath),
    migrationFiles: fileNodes.filter((n) => /(^|\/)migrations?\//.test(clean(n.filePath).toLowerCase())).map((n) => n.filePath),
    dataModelFiles: fileNodes.filter((n) => (n.tags || []).some((t) => /data-model|model-training|data-pipeline/.test(t)) || /(^|\/)(models?|data)(\/|\.)/.test(clean(n.filePath).toLowerCase())).map((n) => n.filePath),
    apiHandlerFiles: fileNodes.filter((n) => (n.tags || []).some((t) => /api-handler|routing/.test(t)) || /(^|\/)(routes|api|controllers|handlers)\//.test(clean(n.filePath).toLowerCase())).map((n) => n.filePath)
  };

  const groupsWithDocs = new Set();
  for (const node of fileNodes) {
    if (node.type !== 'document' && !/\.(md|rst)$/i.test(clean(node.filePath))) continue;
    const group = groupById.get(node.id);
    groupsWithDocs.add(group);
    const text = `${node.summary || ''} ${(node.tags || []).join(' ')}`.toLowerCase();
    for (const candidate of Object.keys(directoryGroups)) if (candidate !== 'root' && text.includes(candidate.toLowerCase())) groupsWithDocs.add(candidate);
  }
  const allGroups = Object.keys(directoryGroups);
  const docCoverage = {
    groupsWithDocs: groupsWithDocs.size,
    totalGroups: allGroups.length,
    coverageRatio: allGroups.length ? groupsWithDocs.size / allGroups.length : 0,
    undocumentedGroups: allGroups.filter((g) => !groupsWithDocs.has(g))
  };

  const dependencyDirection = [];
  const seenPairs = new Set();
  for (const item of interGroupImports) {
    const pair = [item.from, item.to].sort().join('\u0000');
    if (seenPairs.has(pair)) continue;
    seenPairs.add(pair);
    const forward = interCounts.get(`${item.from}\u0000${item.to}`) || 0;
    const reverse = interCounts.get(`${item.to}\u0000${item.from}`) || 0;
    if (forward > reverse) dependencyDirection.push({ dependent: item.from, dependsOn: item.to });
    else if (reverse > forward) dependencyDirection.push({ dependent: item.to, dependsOn: item.from });
  }

  const result = {
    scriptCompleted: true,
    commonPathPrefix: prefix.join('/'),
    directoryGroups,
    nodeTypeGroups,
    importAdjacency: Object.fromEntries(Object.entries(adjacencySets).map(([id, set]) => [id, [...set]])),
    groupImportsFrom,
    groupImportedBy,
    crossCategoryEdges,
    nonCodeConnections,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats: {
      totalFileNodes: fileNodes.length,
      filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([g, ids]) => [g, ids.length])),
      nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([t, ids]) => [t, ids.length]))
    },
    fileFanIn,
    fileFanOut
  };
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n');
} catch (error) {
  fatal(error.stack || error.message || error);
}
