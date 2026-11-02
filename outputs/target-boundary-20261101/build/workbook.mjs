import fs from 'node:fs/promises';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB';
const out=root+'/outputs/target-boundary-20261101';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/workbook-before.xlsx'));
const sheet=wb.worksheets.getItemAt(0);
if(process.argv.includes('--inspect')){
  console.log((await wb.inspect({kind:'workbook,sheet,table',maxChars:2800,tableMaxRows:3,tableMaxCols:13})).ndjson);
  console.log((await wb.inspect({kind:'table',range:sheet.name+'!A5:M18',maxChars:5500,tableMaxRows:14,tableMaxCols:13})).ndjson);
  console.log(wb.help('range.insert',{include:'index,examples,notes',maxChars:2500}).ndjson);
  await fs.writeFile(out+'/source-values.json',JSON.stringify(sheet.getUsedRange().values));
  const preview=await wb.render({sheetName:sheet.name,range:'A13:M18',scale:1});
  await fs.writeFile(out+'/before.png',new Uint8Array(await preview.arrayBuffer()));
}else{
  const verified=JSON.parse(await fs.readFile(root+'/docs/research/tables/target-boundary-2026-11-01/verification.json','utf8'));
  if(!verified.verified || verified.metric_rows!==336)throw Error('Unverified results');
  const source=sheet.getUsedRange().values;
  const updates=JSON.parse(await fs.readFile(out+'/updates.json','utf8'));
  if(updates.length!==4 || source[16][0]!=='Original synthetic anomalies')throw Error('Unexpected template');
  // Move the existing notes as a block before extending the native table.
  for(let row=49;row>=41;row--){
    sheet.getRange(`A${row+4}:M${row+4}`).copyFrom(sheet.getRange(`A${row}:M${row}`),'all');
    sheet.getRange(`A${row+4}:M${row+4}`).format.rowHeight=([44,45].includes(row)?22:row===49?24:32);
  }
  sheet.getRange('A41:M44').clear({applyTo:'all'});
  sheet.tables.items[0].rows.add(null,updates);
  // Preserve all24 historical rows, their values and their cell formats.
  for(let row=40;row>=17;row--){
    sheet.getRange(`A${row+4}:M${row+4}`).copyFrom(sheet.getRange(`A${row}:M${row}`),'all');
    sheet.getRange(`A${row+4}:M${row+4}`).format.rowHeight=32;
  }
  for(let row=17;row<=20;row++){
    sheet.getRange(`A${row}:M${row}`).copyFrom(sheet.getRange('A16:M16'),'all');
    sheet.getRange(`A${row}:M${row}`).format.rowHeight=64;
    sheet.getRange(`A${row}:M${row}`).format.verticalAlignment='center';
  }
  sheet.getRange('A17:M20').values=updates;
  sheet.getRange('A17:A20').format.wrapText=true;
  sheet.getRange('M17:M20').format.wrapText=true;
  sheet.getRange('A4').values=[['One row per technique or scoring pipeline. Original methods and all historical results are preserved.']];
  sheet.getRange('A52').values=[['1 November study: four post-processing families retained boundary rules. Full annual metrics and event IDs are in the target-boundary report.']];
  wb.recalculate();
  const after=sheet.getUsedRange().values;
  for(let i=5;i<40;i++){
    const destination=i<16?i:i+4;
    if(JSON.stringify(source[i])!==JSON.stringify(after[destination]))throw Error('Original row changed: '+(i+1));
  }
  let historicalSeen=false;
  for(const row of after.slice(5,44)){
    if(String(row[4]).includes('2007–2013')){if(historicalSeen)throw Error('Assessment order violated');}
    else historicalSeen=true;
  }
  if(new Set(after.slice(5,44).map(r=>r[0])).size!==39)throw Error('Technique duplication');
  console.log((await wb.inspect({kind:'table',range:sheet.name+'!H17:M20',tableMaxRows:4,tableMaxCols:6,maxChars:2200})).ndjson);
  console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20}})).ndjson);
  for(const [name,range] of [['after','A15:M21'],['notes','A46:M53']]){
    const preview=await wb.render({sheetName:sheet.name,range,scale:1});
    await fs.writeFile(out+'/'+name+'.png',new Uint8Array(await preview.arrayBuffer()));
  }
  await fs.mkdir(out+'/deliverables',{recursive:true});
  await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_full_and_historical.xlsx');
  await fs.writeFile(out+'/workbook-checks.json',JSON.stringify({originalRowsPreserved:35,historicalRowsPreserved:24,newPipelineRows:4,totalRows:39,all2007_2013First:true,uniqueTechniqueNames:true},null,2));
}
