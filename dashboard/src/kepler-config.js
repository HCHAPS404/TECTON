export const MAP_ID='territorio',DATA_ID='municipios';
export const OFFLINE={id:'tecton-local',label:'Fondo local',colorMode:'light',style:{version:8,name:'TECTON local',sources:{},layers:[{id:'background',type:'background',paint:{'background-color':'#eaf0ee'}}]},layerGroups:[]};
export const PALETTES={prob_evento:['#e8f1eb','#c3ddd2','#94c4b1','#61a48d','#31816e','#155a4c'],personas_desplazadas_estimadas:['#fbf0df','#f1d7ad','#dfb47a','#cd8d4d','#b4672e','#88431d'],amplitud:['#eeedf6','#d7d3ea','#b8b0d5','#968bbc','#75669f','#53447c']};

// Contrato explícito: la inferencia automática de Kepler convertiría 05001 en 5001.
export function mapDataset(geojson){
  const definitions=[['_geojson','geojson','GEOMETRY'],['DIVIPOLA','string','STRING'],['municipio','string','STRING'],['departamento','string','STRING'],...['prob_evento','personas_desplazadas_estimadas','personas_desplazadas_q10','personas_desplazadas_q90','amplitud','log_personas','log_amplitud'].map(n=>[n,'real','FLOAT'])];
  return {fields:definitions.map(([name,type,analyzerType],fieldIdx)=>({name,id:name,displayName:name,type,analyzerType,format:'',fieldIdx})),rows:geojson.features.map(f=>definitions.map(([name])=>name==='_geojson'?f:f.properties[name]))};
}
export function mapConfig(metric,previous={}){
  const colorField=metric==='prob_evento'?'prob_evento':metric==='amplitud'?'log_amplitud':'log_personas';
  return {version:'v1',config:{
    visState:{filters:[],layers:[{id:'riesgo',type:'geojson',config:{dataId:DATA_ID,label:'Municipios',color:[25,108,99],isVisible:true,columns:{geojson:'_geojson'},visConfig:{opacity:.9,strokeOpacity:.75,thickness:.5,filled:true,stroked:true,enable3d:false,wireframe:false,strokeColor:[255,255,255],colorRange:{name:'TECTON',type:'sequential',category:'Custom',colors:PALETTES[metric]}},hidden:false},visualChannels:{colorField:{name:colorField,type:'real'},colorScale:'quantize',strokeColorField:null,strokeColorScale:'quantile',sizeField:null,sizeScale:'linear',heightField:null,heightScale:'linear',radiusField:null,radiusScale:'linear'}}],interactionConfig:{tooltip:{fieldsToShow:{[DATA_ID]:['DIVIPOLA','municipio','prob_evento','personas_desplazadas_estimadas','personas_desplazadas_q10','personas_desplazadas_q90'].map(name=>({name,format:name==='prob_evento'?'.1%':null}))},enabled:true},brush:{enabled:false},geocoder:{enabled:false},coordinate:{enabled:false}}},
    mapState:{latitude:previous.latitude??4.45,longitude:previous.longitude??-73.6,zoom:previous.zoom??4.4,pitch:0,bearing:0,dragRotate:false},
    mapStyle:{styleType:OFFLINE.id,visibleLayerGroups:{},topLayerGroups:{},mapStyles:{[OFFLINE.id]:OFFLINE}},
  }};
}
