const fs = require('fs');
const path = require('path');

const root = process.argv[2];
const ua = path.join(root, '.ua');
const scan = JSON.parse(fs.readFileSync(path.join(ua, 'intermediate', 'scan-result.json'), 'utf8'));
fs.writeFileSync(path.join(ua, 'meta.json'), JSON.stringify({
  lastAnalyzedAt: new Date().toISOString(),
  gitCommitHash: 'unavailable-not-a-git-repository',
  version: '1.0.0',
  analyzedFiles: scan.files.length,
}, null, 2));
