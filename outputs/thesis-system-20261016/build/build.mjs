import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const base='/home/k_faz/projects/ESA-ADB/outputs/thesis-system-20261016';
const d=JSON.parse(await fs.readFile(`${base}/build/data.json`,'utf8'));
const wb=Workbook.create(),s=wb.worksheets.add('Thesis results');s.showGridLines=false;
s.getRange('A1:G36').format.font={name:'Arial',size:11,color:'#20313D'};
s.getRange('A1:G36').format.rowHeight=28;s.getRange('A1:G36').format.verticalAlignment='center';
for(const [i,w] of [235,115,110,120,125,100,140].entries())s.getRangeByIndexes(0,i,36,1).format.columnWidthPx=w;
const text=(row,t)=>s.getRange(`A${row}`).values=[[t]];
text(2,'Satellite anomaly detection — system-level comparison');s.getRange('A2').format.font={name:'Arial',size:16,bold:true};
text(3,'Shared data: train 2000–2005; calibration 2006; test 2007–2013. Inputs: channels 40–47.');
text(4,'Shared budget: 40 epochs, all nominal training windows, batch 64. One completed run per technique, seed 42.');
text(5,'System evaluation includes Anomaly and Rare Event IDs from all 76 channels. Both runs are complete and verified.');
text(7,'Pooled results — continuous 2007–2013 assessment');s.getRange('A7').format.font.bold=true;
const header=(row,vals)=>{s.getRange(`A${row}:G${row}`).values=[vals];s.getRange(`A${row}:G${row}`).format={fill:'#243D50',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:35,horizontalAlignment:'center'};};
header(8,['Technique','Inputs','Precision','TPR (event)','FPR (time)','F0.5','Status']);
const m=d.rows[0];s.getRange('A9:G10').values=[['Baseline Transformer AE','40–47',m.precision,m.tpr,m.fpr,null,'Complete'],['Remove short rolls','40–47','Pending','Pending','Pending','Pending','In progress']];
s.getRange('F9').formulas=[['=IF(0.25*C9+D9=0,0,1.25*C9*D9/(0.25*C9+D9))']];
text(12,'TPR (event) = detected target event IDs / total target event IDs. This is event recall.');
text(13,'FPR (time) = nominal false-alarm duration / total nominal duration, excluding annotated intervals.');
text(14,'FPR is duration-based, not pointwise FP/(FP+TN). It uses a different unit from event TPR.');
text(15,'Earlier development experiments used different periods or budgets and are excluded from this comparison.');
text(16,'This compares complete augmentation methods; the historical baseline is not a short-roll-only ablation control.');
text(17,'The benchmark was previously inspected. The pooled F0.5 is rescored over seven years, not averaged by year.');
s.getRange('A12:A17').format.font={name:'Arial',size:10,color:'#536574'};
text(19,'Yearly system results — same frozen model and threshold across all years');s.getRange('A19').format.font.bold=true;
header(20,['Technique','Test year','Precision','TPR (event)','FPR (time)','F0.5','Status']);
for(let i=0;i<7;i++){
 const r=21+2*i,v=d.rows[i+1];s.getRange(`A${r}:G${r+1}`).values=[['Baseline Transformer AE',Number(v.period),v.precision,v.tpr,v.fpr,null,'Complete'],['Remove short rolls',Number(v.period),'Pending','Pending','Pending','Pending','In progress']];
 s.getRange(`F${r}`).formulas=[[`=IF(0.25*C${r}+D${r}=0,0,1.25*C${r}*D${r}/(0.25*C${r}+D${r}))`]];
}
for(const [lo,hi] of [[9,10],[21,34]]){
 s.getRange(`C${lo}:D${hi}`).setNumberFormat('0.0000');s.getRange(`E${lo}:E${hi}`).setNumberFormat('0.0000%');s.getRange(`F${lo}:F${hi}`).setNumberFormat('0.0000');s.getRange(`C${lo}:F${hi}`).format.horizontalAlignment='right';
 for(let row=lo;row<=hi;row++){s.getRange(`A${row}:G${row}`).format.fill=row%2?'#EDF3F7':'#FFFFFF';}
 s.getRange(`G${lo}:G${hi}`).conditionalFormats.add('containsText',{text:'In progress',format:{fill:'#FFF1D6',font:{color:'#805800'}}});
}
s.getRange('F9:F10').format.font.bold=true;s.getRange('B21:B34').setNumberFormat('0');
for(const cell of ['A9','A21'])wb.notes.add({id:`source-${cell}`,target:{cell:{sheetName:s.name,sheetId:s.sheetId,address:cell}},authorId:'',createdAt:'',body:{plainText:'Source: ESA-ADB, long-baseline results, 24 September 2026; source commit 523b0a4. Pooled and all annual metrics independently replayed from the original prediction mask and raw annotation IDs on 16 October 2026. https://github.com/kamyarfaz/ESA-ADB/blob/523b0a4/docs/research/long-baseline-results-2026-09-24.md'}});
if(!d.candidate_verified)throw Error('Candidate verification required');
for(const [i,row] of [10,22,24,26,28,30,32,34].entries()){
 const v=d.candidate_rows[i];s.getRange(`C${row}:G${row}`).values=[[v.precision,v.tpr,v.fpr,null,'Complete']];
 s.getRange(`F${row}`).formulas=[[`=IF(0.25*C${row}+D${row}=0,0,1.25*C${row}*D${row}/(0.25*C${row}+D${row}))`]];
}
for(const cell of ['A10','A22'])wb.notes.add({id:`source-${cell}`,target:{cell:{sheetName:s.name,sheetId:s.sheetId,address:cell}},authorId:'',createdAt:'',body:{plainText:'Source: docs/research/full-short-no-roll-results-2026-10-16.md and tables/full-short-no-roll-2026-10-16/verified_results.csv. All annual and pooled metrics independently replayed from saved predictions and raw annotation IDs on 16 October 2026.'}});
wb.recalculate();
for(const [i,row] of [10,22,24,26,28,30,32,34].entries()){
 if(Math.abs(s.getRange(`F${row}`).values[0][0]-d.candidate_rows[i].f05)>1e-10)throw Error('Candidate formula differs from corrected evaluator');
}
for(const [i,row] of [9,21,23,25,27,29,31,33].entries()){
 const val=s.getRange(`F${row}`).values[0][0];if(Math.abs(val-d.rows[i].f05)>1e-10)throw Error('F0.5 formula differs from corrected evaluator');
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:2000});
await fs.writeFile(`${base}/build/check.ndjson`,errors.ndjson);
console.log((await wb.inspect({kind:'table',range:'Thesis results!A8:G10',include:'values,formulas',tableMaxRows:4,tableMaxCols:7,maxChars:2500})).ndjson);
for(const [name,range] of [['pooled','A1:G17'],['yearly','A19:G34']]){
 const img=await wb.render({sheetName:s.name,range,scale:1,format:'png'});await fs.writeFile(`${base}/build/${name}.png`,new Uint8Array(await img.arrayBuffer()));
}
await fs.mkdir(`${base}/deliverables`,{recursive:true});await(await SpreadsheetFile.exportXlsx(wb)).save(`${base}/deliverables/ESA_thesis_results.xlsx`);
console.log('Exported workbook; all sixteen F0.5 formulas match independent replay.');
