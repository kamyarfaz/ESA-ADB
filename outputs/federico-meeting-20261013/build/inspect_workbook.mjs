import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const p='/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20260930/deliverables/ESA_all_methods_results_2026-09-30.xlsx';
const w=await SpreadsheetFile.importXlsx(await FileBlob.load(p));
console.log((await w.inspect({kind:'sheet',include:'id,name',maxChars:5000})).ndjson);
for(const name of ['Read me','No short rolls']){
 console.log(name,JSON.stringify(w.worksheets.getItem(name).getRange('A1:G8').values));
 const b=await w.render({sheetName:name,range:name==='Read me'?'A1:B5':'A1:G29',scale:1,format:'png'});
 await fs.writeFile('before-'+name.replaceAll(' ','_')+'.png',new Uint8Array(await b.arrayBuffer()));
}
await fs.writeFile('original-cells.json',JSON.stringify(w.worksheets.items.map(s=>({name:s.name,values:s.getUsedRange().values,formulas:s.getUsedRange().formulas,charts:s.charts.items.length}))));
