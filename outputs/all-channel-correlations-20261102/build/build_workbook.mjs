import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const outputDir=path.resolve('outputs/all-channel-correlations-20261102');
const data=JSON.parse(await fs.readFile(path.join(outputDir,'workbook-data.json'),'utf8'));
const wb=Workbook.create();
const navy='#16324F', blue='#1D5C88', teal='#147D83', gray='#EEF1F4';
const short=c=>Number(String(c).replace('channel_',''));
function base(name,color){
  const s=wb.worksheets.add(name);s.showGridLines=false;s.tabColor=color;return s;
}
function style(s,rows,cols){
  const r=s.getRangeByIndexes(0,0,rows,cols);
  r.format.font={name:'Arial',size:10,color:'#17202A'};
  r.format.verticalAlignment='center';r.format.rowHeight=21;
}
function heading(s,title,scope,note){
  s.getRange('A1').values=[[title]];s.getRange('A1').format.font={name:'Arial',size:14,bold:true,color:navy};
  s.getRange('A1').format.rowHeight=29;
  s.getRange('A2').values=[[scope]];s.getRange('A3').values=[[note]];
  s.getRange('A2:A3').format.font={name:'Arial',size:10,italic:true,color:'#425466'};
}
function header(range){range.format.fill=navy;range.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
  range.format.wrapText=true;range.format.horizontalAlignment='center';range.format.rowHeight=32;}
function table(s,row,headers,values,name){
  s.getRangeByIndexes(row,0,1,headers.length).values=[headers];
  if(values.length)s.getRangeByIndexes(row+1,0,values.length,headers.length).values=values;
  const t=s.tables.add(`A${row+1}:${col(headers.length)}${row+1+values.length}`,true,name);
  t.style='TableStyleMedium2';t.showFilterButton=true;
  header(s.getRangeByIndexes(row,0,1,headers.length));return t;
}
function col(n){let s='';for(;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
const groupSheet=base('Groups and guide',teal);
style(groupSheet,45+data.groups.length,10);
groupSheet.getRange('A1:J50').format.columnWidth=14;
groupSheet.getRange('B1:B100').format.columnWidth=19;
groupSheet.getRange('C1:C100').format.columnWidth=43;
heading(groupSheet,'ESA channel correlations','Mission 1: 2000–2005. Mission 2: 2000. Training data only.','Each mission is analyzed separately. These are not cross-mission correlations.');
const overview=[
  ['Mission','Channels','Unique pairs','Pooled |r| ≥ 0.90','Stable high pairs','Stable target pairs'],
  [1,76,data.counts['1'].all_pairs,data.counts['1'].high_pooled,data.counts['1'].stable,data.counts['1'].stable_targets],
  [2,100,data.counts['2'].all_pairs,data.counts['2'].high_pooled,data.counts['2'].stable,data.counts['2'].stable_targets]
];
groupSheet.getRange('A5:F7').values=overview;header(groupSheet.getRange('A5:F5'));groupSheet.getRange('A5:F5').format.rowHeight=42;
const notes=[
 'r near +1: channels move together. r near −1: channels move in opposite directions.',
 'r near 0: weak linear relationship. Nonlinear or delayed relationships may still exist.',
 'Stable high: |Pearson r| ≥ 0.90, same sign, in every training quarter; at least 500 pairs per quarter.',
 'Changes: correlations of adjacent five-minute differences. Strong levels can reflect a shared slow trend.',
 'Gray / n.a.: categorical, constant or fewer than 100 paired samples. Undefined does not mean zero.',
 'Five-minute causal sampling; maximum observation age 10 minutes; annotated intervals excluded per channel.',
 'Counters remain in level matrices and are flagged in the pair list. Correlation alone does not establish anomaly utility.',
 'Complete-linkage groups: every pair qualifies. This partition does not list every overlapping possible group.',
 'All matrices and pair tables are a static analysis snapshot. The source time series are not embedded.',
 'Stability spans 24 quarters for Mission 1 and 4 for Mission 2. These counts do not compare equal durations.'
];
groupSheet.getRange('A9:A18').values=notes.map(x=>[x]);groupSheet.getRange('A9:A18').format.rowHeight=22;
groupSheet.getRange('A19').values=[['Stable high-correlation groups']];groupSheet.getRange('A19').format.font={size:12,bold:true,color:navy};
const groups=[...data.groups].sort((a,b)=>a.mission-b.mission||a.scope.localeCompare(b.scope)||b.size-a.size);
const groupRows=groups.map(g=>[g.mission,g.scope,g.members,g.size,g.min_abs_quarter_r,g.pooled_r_min,g.pooled_r_max,g.change_r_min??'n.a.',g.change_r_max??'n.a.',g.contains_counter?'Yes':'No']);
table(groupSheet,20,['Mission','Group scope','Channel numbers','Size','Lowest |r| across quarters','Pooled r min','Pooled r max','Change r min','Change r max','Contains counters'],groupRows,'CorrelationGroups');
groupSheet.getRange('A21:J21').format.rowHeight=55;
groupSheet.getRangeByIndexes(21,4,groupRows.length,5).setNumberFormat('0.0000');
groupSheet.getRangeByIndexes(21,4,groupRows.length,5).format.horizontalAlignment='right';
for(const c of [0,3,9])groupSheet.getRangeByIndexes(21,c,groupRows.length,1).format.horizontalAlignment='center';
groupSheet.getRangeByIndexes(21,2,groupRows.length,1).format.wrapText=true;
groupSheet.getRangeByIndexes(21,0,groupRows.length,10).format.rowHeight=32;

const renders=[['Groups and guide',`A1:J${21+groupRows.length}`,'guide']];
for(const item of data.matrices){
  const n=item.channels.length;
  const label=item.kind==='pearson'?'Levels':item.kind==='delta'?'Changes':'Spearman';
  const s=base(`M${item.mission} ${label}`,item.mission===1?blue:teal);
  style(s,n+5,n+1);s.getRangeByIndexes(0,0,n+5,n+1).format.columnWidth=8;
  s.getRangeByIndexes(0,0,n+5,1).format.columnWidth=17;
  heading(s,`M${item.mission} ${label}`,`${item.title}. ${n} channels. Training ${data.periods[item.mission]}.`,
    'Blue = negative. White = near zero. Red = positive. Gray / n.a. = undefined.');
  s.getRangeByIndexes(4,0,1,n+1).values=[['Channel',...item.channels.map(Number)]];
  s.getRangeByIndexes(5,0,n,1).values=item.channels.map(x=>[Number(x)]);
  const values=item.values.map(row=>row.map(v=>v===null?'n.a.':v));
  const grid=s.getRangeByIndexes(5,1,n,n);grid.values=values;grid.setNumberFormat('0.000');
  grid.format.horizontalAlignment='right';
  grid.conditionalFormats.add('colorScale',{colors:['#A8CFF0','#FFFFFF','#F5B2AD'],thresholds:[-1,0,1]});
  grid.conditionalFormats.add('containsText',{text:'n.a.',format:{fill:'#D1D5DB',font:{color:'#556270'}}});
  header(s.getRangeByIndexes(4,0,1,n+1));
  s.getRangeByIndexes(5,0,n,1).format.fill='#E8EEF4';s.getRangeByIndexes(5,0,n,1).format.font={bold:true};
  s.freezePanes.freezeRows(5);s.freezePanes.freezeColumns(1);
  renders.push([s.name,'A1:M17',`m${item.mission}-${item.kind}`]);
}
const pairSheet=base('High correlation pairs',blue);
const pairRows=data.pairs.map(r=>[r.mission,short(r.a),short(r.b),r.pearson,r.spearman,r.delta_pearson??'n.a.',
 r.min_abs_quarter_pearson??'n.a.',r.stable_high?'Yes':'No',r.stable_changes?'Yes':'No',r.both_targets?'Yes':'No',r.counter_pair?'Yes':'No',r.pooled_pairs,r.min_quarter_pairs]);
style(pairSheet,pairRows.length+5,13);pairSheet.getRangeByIndexes(0,0,pairRows.length+5,13).format.columnWidth=14;
pairSheet.getRange('A1:A20').format.columnWidth=18;
heading(pairSheet,'High correlation pairs','All unique pairs with pooled |Pearson r| ≥ 0.90. Filter Stable high to retain quarter-stable relationships.',
 'Target pairs and counters have separate flags. Negative r means an inverse relationship.');
table(pairSheet,4,['Mission','Channel A','Channel B','Pearson r','Spearman r','Change r','Lowest quarterly |r|','Stable high','Stable changes','Both targets','Any counter','Pooled pairs','Min quarterly pairs'],pairRows,'HighCorrelationPairs');
pairSheet.getRange('A5:M5').format.rowHeight=42;
pairSheet.getRangeByIndexes(5,3,pairRows.length,4).setNumberFormat('0.0000');
pairSheet.getRangeByIndexes(5,3,pairRows.length,4).format.horizontalAlignment='right';
pairSheet.getRangeByIndexes(5,7,pairRows.length,4).format.horizontalAlignment='center';
pairSheet.getRangeByIndexes(5,11,pairRows.length,2).setNumberFormat('#,##0');
pairSheet.freezePanes.freezeRows(5);pairSheet.freezePanes.freezeColumns(3);
renders.push([pairSheet.name,'A1:M18','pairs']);
const channelSheet=base('Channels',gray);
const channelRows=data.inventory.map(r=>[r.mission,short(r.channel),r.Target,r.Subsystem,r['Physical Unit'],r.Group,
 r.known_counter?'Yes':'No',r.constant?'Yes':'No',r.coverage,r.eligible_samples,r.status]);
style(channelSheet,channelRows.length+10,11);
channelSheet.getRangeByIndexes(0,0,channelRows.length+10,11).format.columnWidth=14;
channelSheet.getRange('D1:F190').format.columnWidth=18;channelSheet.getRange('K1:K190').format.columnWidth=31;
heading(channelSheet,'Channel inventory','All telemetry channel IDs, roles, coverage and Pearson eligibility.',
 'Source: ESA-AD v1.0, https://zenodo.org/records/12528696. Full processing protocol is included in the report.');
table(channelSheet,4,['Mission','Channel','Target','Subsystem','Physical unit','Group','Known counter','Constant','Eligible coverage','Eligible samples','Status'],channelRows,'MissionChannels');
channelSheet.getRange('A5:K5').format.rowHeight=42;
channelSheet.getRangeByIndexes(5,8,channelRows.length,1).setNumberFormat('0.00%');
channelSheet.getRangeByIndexes(5,9,channelRows.length,1).setNumberFormat('#,##0');
channelSheet.freezePanes.freezeRows(5);channelSheet.freezePanes.freezeColumns(2);
renders.push([channelSheet.name,'A1:K20','channels']);
wb.recalculate();
await fs.mkdir(path.join(outputDir,'previews'),{recursive:true});
for(const [sheetName,range,file] of renders){
  const preview=await wb.render({sheetName,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(outputDir,'previews',file+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:20},summary:'Formula error check'});
await fs.writeFile(path.join(outputDir,'workbook-inspection.txt'),errors.ndjson);
const summary=await wb.inspect({kind:'table',range:"'High correlation pairs'!A5:M10",include:'values,formulas',tableMaxRows:6,tableMaxCols:13,maxChars:2500});
await fs.writeFile(path.join(outputDir,'workbook-spot-check.txt'),summary.ndjson);
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(outputDir,'ESA_Mission1_Mission2_correlation_matrices.xlsx'));
console.log(JSON.stringify({sheets:9,highPairs:pairRows.length,groups:groupRows.length,channels:channelRows.length,exported:true}));
