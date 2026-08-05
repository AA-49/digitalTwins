const fs = require('fs');
const path = require('path');

const root = process.argv[2];
const ua = path.join(root, '.ua');
const inter = path.join(ua, 'intermediate');
const scan = JSON.parse(fs.readFileSync(path.join(inter, 'scan-result.json'), 'utf8'));
const assembled = JSON.parse(fs.readFileSync(path.join(inter, 'assembled-graph.json'), 'utf8'));
const layersRaw = JSON.parse(fs.readFileSync(path.join(inter, 'layers.json'), 'utf8'));
const tourRaw = JSON.parse(fs.readFileSync(path.join(inter, 'tour.json'), 'utf8'));
const layers = Array.isArray(layersRaw) ? layersRaw : layersRaw.layers;
const tour = (Array.isArray(tourRaw) ? tourRaw : (tourRaw.steps || tourRaw.tour || []))
  .map((step) => ({
    order: step.order,
    title: step.title,
    description: step.description || step.whyItMatters,
    nodeIds: step.nodeIds || step.nodesToInspect || [],
    ...(step.languageLesson ? { languageLesson: step.languageLesson } : {}),
  }))
  .sort((a, b) => a.order - b.order);

const graph = {
  version: '1.0.0',
  project: {
    name: scan.name,
    languages: scan.languages,
    frameworks: scan.frameworks,
    description: scan.description,
    analyzedAt: new Date().toISOString(),
    gitCommitHash: 'unavailable-not-a-git-repository',
  },
  nodes: assembled.nodes || [],
  edges: assembled.edges || [],
  layers,
  tour,
};

fs.writeFileSync(path.join(inter, 'assembled-graph.json'), JSON.stringify(graph, null, 2));
