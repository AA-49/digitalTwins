const fs = require('fs');
const path = require('path');

const root = process.argv[2];
const ua = path.join(root, '.ua');
const inter = path.join(ua, 'intermediate');
const graph = JSON.parse(fs.readFileSync(path.join(inter, 'assembled-graph.json'), 'utf8'));
const scan = JSON.parse(fs.readFileSync(path.join(inter, 'scan-result.json'), 'utf8'));

fs.writeFileSync(path.join(ua, 'knowledge-graph.json'), JSON.stringify(graph, null, 2));
fs.writeFileSync(path.join(inter, 'fingerprint-input.json'), JSON.stringify({
  projectRoot: root,
  sourceFilePaths: scan.files.map((file) => file.path),
  gitCommitHash: 'unavailable-not-a-git-repository',
}, null, 2));
