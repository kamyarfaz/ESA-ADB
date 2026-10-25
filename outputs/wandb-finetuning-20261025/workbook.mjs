import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB';
const out=root+'/outputs/wandb-finetuning-20261025';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(root+'/ESA_thesis_results_full_and_historical.xlsx'));
console.log((await wb.inspect({kind:'sheet',include:'id,name'})).ndjson);
const sheet=wb.worksheets.getItemAt(0);
await fs.writeFile(out+'/source-values.json',JSON.stringify(sheet.getUsedRange().values));
console.log((await wb.inspect({kind:'table',range:sheet.name+'!A1:O9',tableMaxCols:15,tableMaxRows:9,maxChars:5000})).ndjson);
const img=await wb.render({sheetName:sheet.name,range:'A1:H8',scale:1});
await fs.writeFile(out+'/before.png',new Uint8Array(await img.arrayBuffer()));
const updates=JSON.parse(await fs.readFile(out+'/updates.json','utf8'));
for(const u of updates){sheet.getRange(`H${u.row}:L${u.row}`).values=[u.metrics];sheet.getRange(`M${u.row}`).values=[[u.status]];sheet.getRange(`G${u.row}`).values=[[u.budget]];}
sheet.getRange('A4').values=[['One row per technique. The three W&B methods show calibration-selected results; historical bests remain descriptive.']];
sheet.getRange('A43').values=[['Runs denotes seeds, not LR trials. Historical multi-run metrics are means. Each W&B method tested 3 rates, one seed.']];
sheet.getRange('A48').values=[['Historical rows preserved. Original controls, all attempts and annual metrics are in the W&B fine-tuning report (25 October 2026).']];
sheet.getRange('A49').values=[['W&B: seed 42, reused benchmark, calibration-only selection. Extra inputs: 14, 21, 29. Short-no-roll LR1e-5 repeats prior work.']];
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:sheet.name+'!G15:M16',tableMaxRows:2,tableMaxCols:7,maxChars:2000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:10}})).ndjson);
for(const [name,range] of [['after','G13:M16'],['notes','A42:M49']]){const b=await wb.render({sheetName:sheet.name,range,scale:1});await fs.writeFile(out+'/'+name+'.png',new Uint8Array(await b.arrayBuffer()));}
await fs.mkdir(out+'/deliverables',{recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_wandb_finetuning.xlsx');
