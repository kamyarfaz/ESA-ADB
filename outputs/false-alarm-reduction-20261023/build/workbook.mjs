import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB';
const out=root+'/outputs/false-alarm-reduction-20261023';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/build/template.xlsx'));
const sheet=wb.worksheets.getItem('Thesis results');
if(process.argv.includes('--inspect')){
 const p=await wb.render({sheetName:sheet.name,range:'A13:M16',scale:1});
 await fs.writeFile(out+'/before.png',new Uint8Array(await p.arrayBuffer()));
 console.log((await wb.inspect({kind:'table',range:"'Thesis results'!A14:M16",tableMaxRows:3,tableMaxCols:13,maxChars:2000})).ndjson);
}else{
 const data=JSON.parse(await fs.readFile(out+'/build/data.json','utf8'));
 const rows=sheet.getRange('A6:M40').values;
 for(const update of data.updates){
  const index=rows.findIndex(r=>r[0]===update.label);
  if(index<0)throw Error('Technique not found: '+update.label);
  const values=update.values.slice();
  values[5]=values[5].startsWith('Refined')?'Refined threshold':values[5].startsWith('Original rule retained')?'Original rule retained':'Original remains best';
  sheet.getRange(`H${index+6}:M${index+6}`).values=[values];
 }
 sheet.getRange('A4').values=[['One row per technique. Best reported system F0.5 is descriptive selection, not independent validation.']];
 wb.recalculate();
 console.log((await wb.inspect({kind:'table',range:"'Thesis results'!A14:M16",tableMaxRows:3,tableMaxCols:13,maxChars:2500})).ndjson);
 console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20}})).ndjson);
 const p=await wb.render({sheetName:sheet.name,range:'A13:M16',scale:1});
 await fs.writeFile(out+'/after.png',new Uint8Array(await p.arrayBuffer()));
 await fs.mkdir(out+'/deliverables',{recursive:true});
 await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_alarm_refinement.xlsx');
 console.log('Export saved');
}
