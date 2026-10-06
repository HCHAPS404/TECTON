import React, {useEffect,useMemo,useRef,useState} from 'react';
import KeplerGl from '@kepler.gl/components';
import keplerGlReducer from '@kepler.gl/reducers';
import {addDataToMap,layerConfigChange,wrapTo} from '@kepler.gl/actions';
import {initApplicationConfig} from '@kepler.gl/utils';
import {createStore,combineReducers,applyMiddleware} from 'redux';
import {Provider,useSelector} from 'react-redux';
import {taskMiddleware} from 'react-palm/tasks';
import 'maplibre-gl/dist/maplibre-gl.css';
import {DATA_ID,MAP_ID,OFFLINE,mapConfig,mapDataset} from './kepler-config.js';

const baseMapLibrary={getMapLib:()=>import('maplibre-gl'),mapLibName:'MapLibre',mapLibUrl:'https://maplibre.org/',mapLibCssClass:'maplibregl',mapLibAttributionCssClass:'maplibregl-ctrl-attrib'};
initApplicationConfig({showReleaseBanner:false,escapeXhtmlForWebpack:false,baseMapLibraryConfig:{maplibre:baseMapLibrary,mapbox:baseMapLibrary}});
const reducer=keplerGlReducer.initialState({
  uiState:{readOnly:true,activeSidePanel:null,currentModal:null,locale:'es'},
  mapStyle:{styleType:OFFLINE.id,mapStyles:{[OFFLINE.id]:OFFLINE},mapStylesReplaceDefault:true},
});
const store=createStore(combineReducers({keplerGl:reducer}),applyMiddleware(taskMiddleware));

function MapContent({geojson,metric,maxValue,onSelect}) {
  const parent=useRef(null); const [size,setSize]=useState({width:800,height:460});
  const [ready,setReady]=useState(false);
  const clicked=useSelector(state=>state.keplerGl[MAP_ID]?.visState.clicked);
  const loaded=useSelector(state=>state.keplerGl[MAP_ID]?.visState.datasets[DATA_ID]);
  useEffect(()=>{const code=clicked?.object?.properties?.DIVIPOLA;if(code)onSelect(code);},[clicked,onSelect]);
  useEffect(()=>{
    const observer=new ResizeObserver(entries=>{const {width,height}=entries[0].contentRect;setSize({width:Math.max(1,Math.floor(width)),height:Math.max(1,Math.floor(height))});});
    observer.observe(parent.current);return()=>observer.disconnect();
  },[]);
  const data=useMemo(()=>mapDataset(geojson),[geojson]);
  useEffect(()=>{
    if(!ready || !geojson.features.length)return;
    const previous=store.getState().keplerGl[MAP_ID]?.mapState;
    const config=mapConfig(metric,previous);
    store.dispatch(wrapTo(MAP_ID,addDataToMap({datasets:{info:{id:DATA_ID,label:'Predicciones del mes'},data},options:{centerMap:false,readOnly:true},config})));
  },[ready,data,geojson,metric]);
  useEffect(()=>{
    // La creación del dataset es asíncrona; fijar el dominio cuando finaliza.
    const layer=store.getState().keplerGl[MAP_ID]?.visState.layers[0];
    if(layer)store.dispatch(wrapTo(MAP_ID,layerConfigChange(layer,{colorDomain:[0,metric==='prob_evento'?1:Math.max(1,Math.log1p(maxValue))]})));
  },[loaded,metric,maxValue]);
  return <div ref={parent} className="kepler-host"><KeplerGl id={MAP_ID} width={size.width} height={size.height} mapboxApiAccessToken="" mapStyles={[OFFLINE]} mapStylesReplaceDefault readOnly onKeplerGlInitialized={()=>setReady(true)} cloudProviders={[]} /></div>;
}
export default function KeplerMap(props){return <Provider store={store}><MapContent {...props}/></Provider>;}
