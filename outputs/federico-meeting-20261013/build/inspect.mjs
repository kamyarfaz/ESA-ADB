import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20260930/deliverables/ESA_research_update_2026-09-30.pptx'));
const x=await p.inspect({select:'$.slides[*]',include:['id','index','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']});
await fs.writeFile('reference-inspect.ndjson',x.ndjson);
console.log('masters',p.masters.items.map(m=>m.id));
console.log('placeholders',p.slides.items.map(s=>s.placeholders.items.length));
for(let i=0;i<p.slides.items.length;i++){
 await fs.writeFile(`reference-${i+1}.png`,new Uint8Array(await (await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));
}
console.log(x.ndjson.slice(0,5500));
