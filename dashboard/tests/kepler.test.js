import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {DATA_ID,MAP_ID,OFFLINE,mapConfig,mapDataset} from '../src/kepler-config.js';
const require=createRequire(import.meta.url);
const {default:keplerGlReducer}=require('@kepler.gl/reducers');
const {addDataToMap,layerConfigChange,registerEntry,wrapTo}=require('@kepler.gl/actions');
const {createStore,combineReducers,applyMiddleware}=require('redux');
const {taskMiddleware}=require('react-palm/tasks');

test('Kepler loads polygons, keeps DIVIPOLA string, replaces monthly data and fixes probability scale',async()=>{
  const reducer=keplerGlReducer.initialState({mapStyle:{styleType:OFFLINE.id,mapStyles:{[OFFLINE.id]:OFFLINE}}});
  const store=createStore(combineReducers({keplerGl:reducer}),applyMiddleware(taskMiddleware));
  store.dispatch(registerEntry({id:MAP_ID,mapStylesReplaceDefault:true}));
  const state=()=>store.getState().keplerGl[MAP_ID];
  for(const probability of [.8,.2]){
    const geo={type:'FeatureCollection',features:[{type:'Feature',properties:{DIVIPOLA:'05001',municipio:'Medellín',departamento:'Antioquia',prob_evento:probability,personas_desplazadas_estimadas:12,personas_desplazadas_q10:0,personas_desplazadas_q90:30,amplitud:30,log_personas:Math.log1p(12),log_amplitud:Math.log1p(30)},geometry:{type:'Polygon',coordinates:[[[-75,6],[-74,6],[-74,7],[-75,6]]]}}]};
    const data=mapDataset(geo);
    store.dispatch(wrapTo(MAP_ID,addDataToMap({datasets:{info:{id:DATA_ID,label:'Municipios'},data},options:{centerMap:false,readOnly:true},config:mapConfig('prob_evento')})));
    await new Promise(resolve=>setTimeout(resolve,100));
    assert.deepEqual(Object.keys(state().visState.datasets),[DATA_ID]);
    const dataset=state().visState.datasets[DATA_ID];
    const codeField=dataset.fields.find(f=>f.name==='DIVIPOLA');
    assert.equal(codeField.type,'string');
    assert.equal(state().visState.layers.length,1);
    const layer=state().visState.layers[0];
    assert.equal(layer.type,'geojson');assert.equal(layer.config.colorField.name,'prob_evento');
    store.dispatch(wrapTo(MAP_ID,layerConfigChange(layer,{colorDomain:[0,1]})));
    assert.deepEqual(state().visState.layers[0].config.colorDomain,[0,1]);
    assert.equal(state().mapStyle.styleType,OFFLINE.id);
    assert.equal(data.rows[0][1],'05001');
    assert.equal(data.rows[0][0].properties.prob_evento,probability);
  }
});
