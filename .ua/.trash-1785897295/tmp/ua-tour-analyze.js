const fs = require('fs');
try {
  const [inputPath, outputPath] = process.argv.slice(2);
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = input.nodes || [], edges = input.edges || [], layers = input.layers || [];
  const byId = new Map(nodes.map(n => [n.id, n]));
  const fanIn = new Map(nodes.map(n => [n.id, 0])), fanOut = new Map(nodes.map(n => [n.id, 0]));
  for (const e of edges) {
    if (fanOut.has(e.source)) fanOut.set(e.source, fanOut.get(e.source) + 1);
    if (fanIn.has(e.target)) fanIn.set(e.target, fanIn.get(e.target) + 1);
  }
  const ranking = (counts, key) => nodes.map(n => ({ id: n.id, [key]: counts.get(n.id), name: n.name }))
    .sort((a,b) => b[key] - a[key] || a.id.localeCompare(b.id)).slice(0,20);
  const fanInRanking = ranking(fanIn, 'fanIn'), fanOutRanking = ranking(fanOut, 'fanOut');
  const outVals = [...fanOut.values()].sort((a,b)=>b-a), inVals = [...fanIn.values()].sort((a,b)=>a-b);
  const highOut = outVals[Math.max(0, Math.ceil(outVals.length * .1)-1)] || 0;
  const lowIn = inVals[Math.max(0, Math.ceil(inVals.length * .25)-1)] || 0;
  const entryNames = new Set(['index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js','mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py','Application.java','Main.java','Program.cs','config.ru','index.php','App.swift','Application.kt','main.cpp','main.c']);
  const candidates = nodes.map(n => {
    const path = String(n.filePath || n.name || '').replace(/\\/g,'/'), name = path.split('/').pop(), depth = path.split('/').filter(Boolean).length;
    let score = 0;
    if (n.type === 'file') {
      if (entryNames.has(name)) score += 3;
      if (depth <= 2) score += 1;
      if ((fanOut.get(n.id)||0) >= highOut) score += 1;
      if ((fanIn.get(n.id)||0) <= lowIn) score += 1;
    } else if (n.type === 'document') {
      if (path === 'README.md') score += 5;
      else if (depth === 1 && name.endsWith('.md')) score += 2;
    }
    return { id:n.id, score, name:n.name, summary:n.summary || '' };
  }).filter(x=>x.score>0).sort((a,b)=>b.score-a.score || a.id.localeCompare(b.id)).slice(0,5);
  const start = candidates.map(x=>byId.get(x.id)).find(n=>n && n.type==='file');
  const adjacency = new Map(nodes.map(n=>[n.id,[]]));
  for (const e of edges) if (['imports','calls'].includes(e.type) && adjacency.has(e.source) && byId.has(e.target)) adjacency.get(e.source).push(e.target);
  for (const targets of adjacency.values()) targets.sort();
  const order=[], depthMap={}, byDepth={};
  if (start) {
    const queue=[start.id]; depthMap[start.id]=0;
    while(queue.length) {
      const id=queue.shift(), depth=depthMap[id]; order.push(id); (byDepth[depth] ||= []).push(id);
      for(const target of adjacency.get(id)||[]) if(!(target in depthMap)) { depthMap[target]=depth+1; queue.push(target); }
    }
  }
  const info=n=>({id:n.id,name:n.name,type:n.type,summary:n.summary||''});
  const nonCodeFiles={
    documentation:nodes.filter(n=>n.type==='document').map(info),
    infrastructure:nodes.filter(n=>['service','pipeline','resource'].includes(n.type)).map(info),
    data:nodes.filter(n=>['table','schema','endpoint'].includes(n.type)).map(info),
    config:nodes.filter(n=>n.type==='config').map(info)
  };
  const mutual=[];
  for(const e of edges) if(['imports','calls'].includes(e.type) && e.source<e.target && edges.some(r=>r.source===e.target&&r.target===e.source&&r.type===e.type)) mutual.push([e.source,e.target]);
  const clusters=[];
  for(const pair of mutual) {
    const set=new Set(pair); let changed=true;
    while(changed && set.size<5) { changed=false; for(const n of nodes) { if(set.has(n.id)) continue; const links=[...set].filter(m=>edges.some(e=>(e.source===n.id&&e.target===m)||(e.source===m&&e.target===n.id))).length; if(links>=2){set.add(n.id);changed=true;if(set.size>=5)break;} } }
    const ids=[...set].sort(), signature=ids.join('|');
    if(!clusters.some(c=>c.signature===signature)) clusters.push({nodes:ids,edgeCount:edges.filter(e=>set.has(e.source)&&set.has(e.target)).length,signature});
  }
  clusters.sort((a,b)=>b.edgeCount-a.edgeCount||a.signature.localeCompare(b.signature));
  const nodeSummaryIndex=Object.fromEntries(nodes.map(n=>[n.id,{name:n.name,type:n.type,summary:n.summary||''}]));
  const result={scriptCompleted:true,entryPointCandidates:candidates,fanInRanking,fanOutRanking,bfsTraversal:{startNode:start?start.id:null,order,depthMap,byDepth},nonCodeFiles,clusters:clusters.slice(0,10).map(({nodes,edgeCount})=>({nodes,edgeCount})),layers:{count:layers.length,list:layers.map(({id,name,description})=>({id,name,description}))},nodeSummaryIndex,totalNodes:nodes.length,totalEdges:edges.length};
  fs.writeFileSync(outputPath,JSON.stringify(result,null,2));
} catch (error) {
  console.error(error.stack || error.message);
  process.exit(1);
}
