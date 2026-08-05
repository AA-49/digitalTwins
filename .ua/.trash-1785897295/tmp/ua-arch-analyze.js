const fs = require('fs');
const path = require('path');
const [inputPath, outputPath] = process.argv.slice(2);
const fail = (error) => { process.stderr.write(String(error.stack || error) + '\n'); process.exit(1); };
if (!inputPath || !outputPath) fail(new Error('Expected input and output JSON paths'));

try {
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = input.fileNodes || [];
  const ids = new Set(nodes.map((node) => node.id));
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const imports = (input.importEdges || []).filter((edge) => ids.has(edge.source) && ids.has(edge.target));
  const allEdges = (input.allEdges || []).filter((edge) => ids.has(edge.source) && ids.has(edge.target));
  const clean = (value) => String(value || '').replace(/\\/g, '/').replace(/^\.\//, '');
  const paths = nodes.map((node) => clean(node.filePath));
  const dirs = paths.map((p) => p.split('/').slice(0, -1));
  const prefix = [];
  if (dirs.length) for (let i = 0; i < Math.min(...dirs.map((d) => d.length)); i++) {
    if (dirs.every((d) => d[i] === dirs[0][i])) prefix.push(dirs[0][i]); else break;
  }
  const rawGroups = paths.map((p) => { const rest = p.split('/').slice(prefix.length); return rest.length > 1 ? rest[0] : 'root'; });
  const flat = new Set(rawGroups).size === 1 && rawGroups[0] === 'root';
  const flatGroup = (node) => {
    const p = clean(node.filePath).toLowerCase(); const base = path.posix.basename(p);
    if (/^(test_.*\.py|.*\.(test|spec)\.)/.test(base)) return 'test';
    if (/dockerfile|docker-compose/.test(base)) return 'infrastructure';
    if (/\.(md|rst)$/.test(base)) return 'documentation';
    if (/\.(ya?ml|json|toml|ini)$/.test(base) || base.startsWith('.env')) return 'config';
    return path.posix.extname(base).slice(1) || 'other';
  };
  const groupFor = (node) => {
    if (flat) return flatGroup(node);
    const rest = clean(node.filePath).split('/').slice(prefix.length);
    return rest.length > 1 ? rest[0] : 'root';
  };
  const directoryGroups = {}, nodeTypeGroups = {}, groupById = new Map();
  for (const node of nodes) {
    const group = groupFor(node); groupById.set(node.id, group);
    (directoryGroups[group] ||= []).push(node.id); (nodeTypeGroups[node.type] ||= []).push(node.id);
  }
  const adjacency = Object.fromEntries(nodes.map((node) => [node.id, []]));
  const fileFanIn = Object.fromEntries(nodes.map((node) => [node.id, 0]));
  const fileFanOut = Object.fromEntries(nodes.map((node) => [node.id, 0]));
  const inter = new Map();
  for (const edge of imports) {
    if (!adjacency[edge.source].includes(edge.target)) adjacency[edge.source].push(edge.target);
    const a = groupById.get(edge.source), b = groupById.get(edge.target);
    if (a !== b) inter.set(`${a}\0${b}`, (inter.get(`${a}\0${b}`) || 0) + 1);
  }
  for (const [source, targets] of Object.entries(adjacency)) { fileFanOut[source] = targets.length; for (const target of targets) fileFanIn[target]++; }
  const groupImportsFrom = Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, []]));
  const groupImportedBy = Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, []]));
  const interGroupImports = [...inter].map(([key, count]) => { const [from, to] = key.split('\0'); groupImportsFrom[from].push(to); groupImportedBy[to].push(from); return { from, to, count }; });
  for (const map of [groupImportsFrom, groupImportedBy]) for (const key of Object.keys(map)) map[key] = [...new Set(map[key])];
  const intraGroupDensity = {};
  for (const group of Object.keys(directoryGroups)) {
    let internalEdges = 0, totalEdges = 0;
    for (const edge of imports) { const a = groupById.get(edge.source), b = groupById.get(edge.target); if (a === group || b === group) totalEdges++; if (a === group && b === group) internalEdges++; }
    intraGroupDensity[group] = { internalEdges, totalEdges, density: totalEdges ? internalEdges / totalEdges : 0 };
  }
  const rules = [
    [/^(routes|api|controllers|endpoints|handlers|blueprints|serializers)$/i, 'api'],
    [/^(services|core|lib|domain|logic|internal|signals|jobs)$/i, 'service'],
    [/^(models|db|data|persistence|repository|entities|migrations|sql|database|schema)$/i, 'data'],
    [/^(components|views|pages|ui|layouts|screens)$/i, 'ui'],
    [/^(middleware|plugins|interceptors|guards)$/i, 'middleware'],
    [/^(utils|helpers|common|shared|tools|pkg)$/i, 'utility'],
    [/^(config|constants|env|settings)$/i, 'config'],
    [/^(__tests__|test|tests|spec|specs)$/i, 'test'],
    [/^(types|interfaces|schemas|contracts|dtos)$/i, 'types'],
    [/^(assets|static|public|templates)$/i, 'assets'],
    [/^(docs|documentation|wiki)$/i, 'documentation'],
    [/^(deploy|deployment|infra|infrastructure|docker|k8s|kubernetes|helm|charts|terraform|tf)$/i, 'infrastructure'],
    [/^(\.github|\.gitlab|\.circleci)$/i, 'ci-cd']
  ];
  const patternMatches = {};
  for (const group of Object.keys(directoryGroups)) { const match = rules.find(([r]) => r.test(group)); if (match) patternMatches[group] = match[1]; }
  const cross = new Map();
  for (const edge of allEdges) { const a = byId.get(edge.source), b = byId.get(edge.target); if (a.type === b.type) continue; const key = `${a.type}\0${b.type}\0${edge.type}`; cross.set(key, (cross.get(key) || 0) + 1); }
  const crossCategoryEdges = [...cross].map(([key, count]) => { const [fromType, toType, edgeType] = key.split('\0'); return { fromType, toType, edgeType, count }; });
  const nonCodeConnections = allEdges.filter((edge) => byId.get(edge.source).type !== 'file' && byId.get(edge.target).type === 'file');
  const low = paths.map((p) => p.toLowerCase());
  const infrastructureNode = (node) => node.type === 'service' || node.type === 'pipeline' || /(^|\/)(dockerfile[^/]*|docker-compose[^/]*\.ya?ml|.*\.tf(vars)?|jenkinsfile|\.gitlab-ci\.ya?ml)$/.test(clean(node.filePath).toLowerCase()) || /(^|\/)(k8s|kubernetes|helm|charts|terraform|\.github\/workflows)(\/|$)/.test(clean(node.filePath).toLowerCase());
  const deploymentTopology = {
    hasDockerfile: low.some((p) => /(^|\/)dockerfile[^/]*$/.test(p)), hasCompose: low.some((p) => /(^|\/)docker-compose[^/]*\.ya?ml$/.test(p)),
    hasK8s: low.some((p) => /(^|\/)(k8s|kubernetes|helm|charts)(\/|$)/.test(p)), hasTerraform: low.some((p) => /\.tf(vars)?$/.test(p)),
    hasCI: low.some((p) => /(^|\/)\.github\/workflows\//.test(p) || /(^|\/)(\.gitlab-ci\.ya?ml|jenkinsfile)$/.test(p)),
    infraFiles: nodes.filter(infrastructureNode).map((node) => node.filePath)
  };
  const dataPipeline = {
    schemaFiles: nodes.filter((n) => n.type === 'schema' || /\.(sql|graphql|gql|proto)$/.test(clean(n.filePath).toLowerCase())).map((n) => n.filePath),
    migrationFiles: nodes.filter((n) => /(^|\/)migrations?\//.test(clean(n.filePath).toLowerCase())).map((n) => n.filePath),
    dataModelFiles: nodes.filter((n) => (n.tags || []).some((t) => /data-model|model-training|model-schema/.test(t))).map((n) => n.filePath),
    apiHandlerFiles: nodes.filter((n) => (n.tags || []).some((t) => /api-handler|routing/.test(t))).map((n) => n.filePath)
  };
  const groupsWithDocs = new Set(nodes.filter((n) => n.type === 'document' || /\.(md|rst)$/i.test(clean(n.filePath))).map((n) => groupById.get(n.id)));
  const allGroups = Object.keys(directoryGroups);
  const docCoverage = { groupsWithDocs: groupsWithDocs.size, totalGroups: allGroups.length, coverageRatio: allGroups.length ? groupsWithDocs.size / allGroups.length : 0, undocumentedGroups: allGroups.filter((g) => !groupsWithDocs.has(g)) };
  const dependencyDirection = []; const pairs = new Set();
  for (const item of interGroupImports) { const pair = [item.from, item.to].sort().join('\0'); if (pairs.has(pair)) continue; pairs.add(pair); const f = inter.get(`${item.from}\0${item.to}`) || 0, r = inter.get(`${item.to}\0${item.from}`) || 0; if (f > r) dependencyDirection.push({ dependent: item.from, dependsOn: item.to }); else if (r > f) dependencyDirection.push({ dependent: item.to, dependsOn: item.from }); }
  const result = { scriptCompleted: true, commonPathPrefix: prefix.join('/'), directoryGroups, nodeTypeGroups, importAdjacency: adjacency, groupImportsFrom, groupImportedBy, crossCategoryEdges, nonCodeConnections, interGroupImports, intraGroupDensity, patternMatches, deploymentTopology, dataPipeline, docCoverage, dependencyDirection, fileStats: { totalFileNodes: nodes.length, filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([g, list]) => [g, list.length])), nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([t, list]) => [t, list.length])) }, fileFanIn, fileFanOut };
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n');
} catch (error) { fail(error); }
