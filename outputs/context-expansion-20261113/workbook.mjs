import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root='/home/k_faz/projects/ESA-ADB';
const out=root+'/outputs/context-expansion-20261113';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(out+'/workbook-before.xlsx'));
console.log((await wb.inspect({kind:'workbook,sheet,table',maxChars:3500,tableMaxRows:3,tableMaxCols:8})).ndjson);
const sheet=wb.worksheets.getItemAt(0);
await fs.writeFile(out+'/source-values.json',JSON.stringify(sheet.getUsedRange().values));
console.log((await wb.inspect({kind:'table',range:sheet.name+'!A1:O9',tableMaxCols:15,tableMaxRows:9,maxChars:5000})).ndjson);
const img=await wb.render({sheetName:sheet.name,range:'A1:M8',scale:1});
await fs.writeFile(out+'/before.png',new Uint8Array(await img.arrayBuffer()));
if(!process.argv.includes('--edit'))process.exit(0);
const data=JSON.parse(await fs.readFile(out+'/workbook-data.json','utf8'));
const study=wb.worksheets.add('Input expansion');
study.showGridLines=false;
study.getRange('A2').values=[['Matched input expansion — 13 November 2026']];
study.getRange('A3').values=[['All three seeds, both arms. Pooled 2007–2013 first; annual results below. Historical results remain on the original sheet.']];
study.getRange('A4').values=[['Train 2000–2005; calibrate 2006 on events on 40–47; assess reused 2007–2013. 40 epochs per run. FPR is a fraction of nominal time.']];
const headers=['Technique','Seed','Arm','Input channels','Assessment','40–47 F0.5','40–47 TPR','40–47 time FPR','40–47 detected / total','System F0.5','System TPR','System time FPR','System detected / total'];
study.getRange('A5:M5').values=[headers];
study.getRange(`A6:M${5+data.rows.length}`).values=data.rows;
study.getRange(`A2:M${5+data.rows.length}`).format.font={name:'Arial',size:10,color:'#203D50'};
study.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:'#203D50'};
study.getRange('A3:A4').format.font={name:'Arial',size:10,color:'#475569'};
study.getRange('A3:A4').format.rowHeight=25;
study.getRange('A5:M5').format={fill:'#203D50',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:43};
study.getRange('A5:A149').format.columnWidth=43;
study.getRange('B5:B149').format.columnWidth=8;
study.getRange('C5:C149').format.columnWidth=13;
study.getRange('D5:D149').format.columnWidth=29;
study.getRange('E5:E149').format.columnWidth=15;
study.getRange('F5:M149').format.columnWidth=15;
study.getRange('I5:I149').format.columnWidth=17;
study.getRange('M5:M149').format.columnWidth=17;
study.getRange('A6:M149').format.rowHeight=28;
study.getRange('A6:M149').format.verticalAlignment='center';
study.getRange('F6:H149').setNumberFormat('0.000000');
study.getRange('J6:L149').setNumberFormat('0.000000');
for(let row=6;row<=149;row+=2)study.getRange(`A${row}:M${row}`).format.fill='#EEF3F6';
study.tables.add('A5:M149',true,'ContextExpansionResults');
study.freezePanes.freezeRows(5);
study.freezePanes.freezeColumns(1);
const notes=[
 'Arm control: fresh matched control. Arm context: add training-selected channels 33 and 49. Seeds 42, 43, 44 are all reported.',
 '40–47 metrics are the primary scope. System metrics cover all anomaly and rare-event IDs. TPR is event-ID recall; FPR is duration based.',
 'All inputs also enter the maximum reconstruction-error score. Additional channels therefore add alarm evidence as well as model inputs.',
 'Masked pretraining: 20 masked epochs plus 20 short-no-roll epochs. Other families: 40 short-no-roll epochs. More inputs require more compute.',
 'The third family retains 14, 21 and 29. Its added-input seeds 43 and 44 have time FPR near 5.9%; event F0.5 alone hides this weakness.',
 'The calibration alarm-fraction cap is 1%. It does not constrain assessment alarms. Only six target calibration IDs inform selection.',
 'Three seeds, reused benchmark and inherited offline preprocessing limit generalization. No robust thesis-target achievement is claimed.',
 'All 288 scope/period metric rows and event IDs were independently verified. Pooled values are recomputed, not averages of yearly metrics.',
 'Source: https://github.com/kamyarfaz/ESA-ADB/blob/main/docs/research/context-expansion-results-2026-11-13.md',
 'Backup: https://github.com/kamyarfaz/ESA-ADB/releases/tag/context-expansion-2026-11-13'
];
for(let i=0;i<notes.length;i++){
 const row=152+i;study.mergeCells(`A${row}:M${row}`);study.getRange(`A${row}`).values=[[notes[i]]];
 study.getRange(`A${row}:M${row}`).format={font:{name:'Arial',size:10,color:'#475569'},rowHeight:25};
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Input expansion!A6:M8',tableMaxCols:13,tableMaxRows:3,maxChars:2200})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:20}})).ndjson);
for(const [name,range] of [['pooled','A2:M23'],['annual','A24:M30'],['notes','A152:M161']]){
 const preview=await wb.render({sheetName:'Input expansion',range,scale:1});
 await fs.writeFile(out+'/'+name+'.png',new Uint8Array(await preview.arrayBuffer()));
}
await fs.mkdir(out+'/deliverables',{recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/deliverables/ESA_thesis_results_full_and_historical.xlsx');
