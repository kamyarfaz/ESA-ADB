import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB';
const out=root+'/outputs/combinations-20261022';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/build/template.xlsx'));
const sheet=wb.worksheets.getItem('Thesis results');
if(process.argv.includes('--inspect')){
 const image=await wb.render({sheetName:sheet.name,range:'A5:M14',scale:1});
 await fs.writeFile(out+'/before.png',new Uint8Array(await image.arrayBuffer()));
 console.log((await wb.inspect({kind:'table',range:"'Thesis results'!A5:M9",tableMaxRows:5,tableMaxCols:13,maxChars:2500})).ndjson);
}else{
 const data=JSON.parse(await fs.readFile(out+'/build/data.json','utf8'));
 const old=sheet.getRange('A6:M37').values;
 const notes=sheet.getRange('A39:M45').values;
 const rows=[...old,...data.rows];
 for(const r of rows) if(r[0]==='MOSAD') {r[12]='Stopped by user. Assessment not completed'; for(let i=7;i<12;i++)r[i]='Not assessed';}
 rows.sort((a,b)=>Number(!String(a[4]).replaceAll('–','-').includes('2007-2013'))-Number(!String(b[4]).replaceAll('–','-').includes('2007-2013')));
 sheet.getRange('A38:M40').copyFrom(sheet.getRange('A35:M37'),'all');
 sheet.getRange('A41:M48').copyFrom(sheet.getRange('A38:M45'),'all');
 sheet.getRange('A6:M48').clear({applyTo:'contents'});
 sheet.getRange('A6:M40').values=rows;
 sheet.getRange('A42:M48').values=notes;
 sheet.getRange('A46').values=[['Not recorded = unavailable metadata. Not assessed = MOSAD stopped by user before assessment.']];
 sheet.getRange('A48').values=[['Historical rows preserved. Full-period masked control and annual metrics are in the accompanying comparison report.']];
 sheet.getRange('A49').values=[['Combinations use seed 42 and a reused benchmark. Fusion uses two models; approach 3 adds channels 14, 21, 29.']];
 sheet.getRange('A49:M49').format.font={name:'Arial',size:10};
 sheet.getRange('A49:M49').format.rowHeight=24;
 sheet.getRange('H14:I16').setNumberFormat('0.0000');
 sheet.getRange('J14:J16').setNumberFormat('0.000000');
 sheet.getRange('K14:L16').setNumberFormat('0.0000');
 sheet.getRange('A13:M16').format.rowHeight=48;
 sheet.getRange('A13:G16').format.wrapText=true;
 sheet.getRange('M13:M16').format.wrapText=true;
 const table=sheet.tables.items[0]; const style=table.style; table.delete();
 const t=sheet.tables.add('A5:M40',true,'FullPeriodAndHistoricalResults'); t.style=style;
 wb.recalculate();
 console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},summary:'Error scan'})).ndjson);
 for(const [name,range] of [['top','A5:M16'],['new','A32:M40']]){
  const image=await wb.render({sheetName:sheet.name,range,scale:1});
  await fs.writeFile(out+'/'+name+'.png',new Uint8Array(await image.arrayBuffer()));
 }
 await fs.mkdir(out+'/deliverables',{recursive:true});
 await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_combinations.xlsx');
 console.log('Saved 35 unique technique rows, all historical results retained.');
}
