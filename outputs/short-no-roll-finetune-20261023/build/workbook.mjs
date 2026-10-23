import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const out='/home/k_faz/projects/ESA-ADB/outputs/short-no-roll-finetune-20261023';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/build/template.xlsx'));
const s=wb.worksheets.getItem('Thesis results');
const rows=s.getRange('A6:M40').values;
console.log(rows.map((r,i)=>[i+6,r[0],r[12]]));
const index=rows.findIndex(r=>String(r[0]).toLowerCase().includes('short')&&!String(r[0]).includes('+'));
if(index<0)throw Error('Missing baseline row');
const row=index+6;
if(!process.argv.includes('--inspect')){
 s.getRange(`M${row}`).values=[['Original retained']];
 wb.recalculate();
 console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20}})).ndjson);
}
const image=await wb.render({sheetName:s.name,range:`A${row}:M${row}`,scale:1});
await fs.writeFile(out+(process.argv.includes('--inspect')?'/before.png':'/after.png'),new Uint8Array(await image.arrayBuffer()));
if(!process.argv.includes('--inspect'))await(await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_short_no_roll_finetuning.xlsx');
