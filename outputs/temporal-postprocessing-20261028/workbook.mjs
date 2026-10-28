import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB',out=root+'/outputs/temporal-postprocessing-20261028';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/workbook-before.xlsx'));
const s=wb.worksheets.getItemAt(0);
console.log((await wb.inspect({kind:'table',range:s.name+'!J14:M16',tableMaxRows:3,tableMaxCols:4})).ndjson);
const img=await wb.render({sheetName:s.name,range:'J14:M16',scale:1});
await fs.writeFile(out+'/before.png',new Uint8Array(await img.arrayBuffer()));

const verified=JSON.parse(await fs.readFile(root+'/docs/research/tables/temporal-postprocessing-2026-10-28/verification.json','utf8'));if(!verified.verified)throw Error('Unverified');
s.getRange('A48').values=[['28 October: temporal processing retained all three W&B alarm rules. Scores unchanged. Full comparison is in the temporal-processing report.']];
wb.recalculate();
const after=await wb.render({sheetName:s.name,range:'A47:M49',scale:1});
await fs.writeFile(out+'/after.png',new Uint8Array(await after.arrayBuffer()));
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!',options:{useRegex:true,maxResults:10}})).ndjson);
await fs.mkdir(out+'/deliverables',{recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_temporal_postprocessing.xlsx');
