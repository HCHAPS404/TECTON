import test from 'node:test';
import assert from 'node:assert/strict';
import {COLUMNS,enrichRows,filterRows,geographyIndex,normalizeGeography,parseCSV,readDataset,selectGeoJSON,summarize,toCSV,validateRows} from '../src/data.js';

const record=(code='05001',fecha='2022-10-01',p=.8)=>({DIVIPOLA:code,fecha,prob_evento:p,personas_desplazadas_estimadas:12,personas_desplazadas_q10:0,personas_desplazadas_q90:30});
const geo={type:'FeatureCollection',features:[{type:'Feature',properties:{mpio_cdpmp:'05001',mpio_cnmbr:'Medellín',dpto_cnmbr:'Antioquia'},geometry:{type:'Polygon',coordinates:[[[-75,6],[-74,6],[-74,7],[-75,6]]]}}]};

test('CSV round-trip preserves leading zeroes and exact official columns',()=>{
  const rows=validateRows([record()]);const csv=toCSV(rows);
  assert.deepEqual(Object.keys(parseCSV(csv)[0]),COLUMNS);
  assert.equal(validateRows(parseCSV('\uFEFF'+csv))[0].DIVIPOLA,'05001');
  assert.equal(validateRows(parseCSV(csv))[0].prob_evento,.8);
});
test('RFC4180 quotation and malformed row detection',()=>{
  const csv=COLUMNS.join(',')+'\r\n"05001","2022-10-01",0.8,12,0,30\r\n';
  assert.equal(parseCSV(csv)[0].DIVIPOLA,'05001');
  for(const bad of ['a,b\n1,2',COLUMNS.join(',')+'\n05001,2022-10-01,0.8,12,0',csv.replace('"05001"','"05001"extra')])assert.throws(()=>parseCSV(bad));
});
test('rejects duplicates, missing numeric values, invalid dates, bounds and intervals',()=>{
  for(const bad of [{DIVIPOLA:'5001'},{fecha:'2022-13-01'},{fecha:'2022-10-02'},{prob_evento:1.2},{prob_evento:''},{prob_evento:null},{prob_evento:true},{personas_desplazadas_estimadas:-1},{personas_desplazadas_q90:NaN},{personas_desplazadas_q10:50}])assert.throws(()=>validateRows([{...record(),...bad}]));
  assert.throws(()=>validateRows([record(),record()]));
});
test('shared filters reconcile map, table and KPIs; missing geometry stays in totals',()=>{
  const index=geographyIndex(normalizeGeography(geo));
  const all=enrichRows(validateRows([record(),record('05002'),record('11001'),record('05001','2022-11-01')]),index);
  const rows=filterRows(all,{month:'2022-10-01',department:'05'});
  assert.equal(rows.length,2);assert.equal(summarize(rows,.5).above,2);assert.equal(summarize(rows,.5).people,24);
  assert.equal(selectGeoJSON(rows,index).features.length,1);
  assert.equal(filterRows(all,{month:'2022-10-01',query:'medellin'}).length,1);
  assert.equal(selectGeoJSON(rows,index).features[0].properties.personas_desplazadas_q90,30);
});
test('geography rejects bad CRS, repeated codes, missing rings and non-polygons',()=>{
  assert.equal(normalizeGeography(geo).features[0].properties.DIVIPOLA,'05001');
  assert.throws(()=>normalizeGeography({...geo,crs:{properties:{name:'EPSG:3857'}}}));
  assert.throws(()=>normalizeGeography({...geo,features:[...geo.features,...geo.features]}));
  assert.throws(()=>normalizeGeography({...geo,features:[{...geo.features[0],geometry:{type:'Point',coordinates:[-75,6]}}]}));
  assert.throws(()=>normalizeGeography({...geo,features:[{...geo.features[0],geometry:{type:'Polygon',coordinates:[[[500000,500000],[1,2],[3,4],[500000,500000]]]}}]}));
});
test('JSON carries explicit synthetic identity; isolated CSV has unknown provenance',()=>{
  assert.equal(readDataset(toCSV([record()]),'predicciones.csv').synthetic,null);
  assert.equal(readDataset(JSON.stringify({schema_version:1,synthetic:true,rows:[record()]}),'datos.json').synthetic,true);
  assert.throws(()=>readDataset(JSON.stringify({schema_version:1,rows:[record()]}),'datos.json'));
});
