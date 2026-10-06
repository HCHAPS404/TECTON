import React from 'react';
import {monthLabel,number,percent} from './data.js';

export default function Trend({rows,probability=false}) {
  if (!rows.length) return <p className="muted">Selecciona un municipio.</p>;
  const sorted = [...rows].sort((a,b)=>a.fecha.localeCompare(b.fecha));
  const W=600,H=150,left=48,right=10,top=15,bottom=32;
  const max = probability ? 1 : Math.max(1,...sorted.map(r=>r.personas_desplazadas_q90),...sorted.map(r=>r.personas_desplazadas_estimadas));
  const x=i=>left+i*(W-left-right)/Math.max(1,sorted.length-1);
  const y=v=>H-bottom-v/max*(H-top-bottom);
  const field=probability?'prob_evento':'personas_desplazadas_estimadas';
  const line=sorted.map((r,i)=>`${x(i)},${y(r[field])}`).join(' ');
  const band=sorted.map((r,i)=>`${x(i)},${y(r.personas_desplazadas_q90)}`).concat(sorted.map((r,i)=>`${x(i)},${y(r.personas_desplazadas_q10)}`).reverse()).join(' ');
  return <figure className="trend"><figcaption>{probability?'Probabilidad de evento':'Personas estimadas e intervalo q10–q90'}</figcaption>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${probability?'Probabilidad':'Personas e intervalo'} por mes. Valores exactos en la tabla de detalle.`}>
      {[0,.5,1].map(t=><g key={t}><line x1={left} x2={W-right} y1={y(max*t)} y2={y(max*t)} stroke="#e1e6e7"/><text x={left-8} y={y(max*t)+4} textAnchor="end">{probability?percent(max*t):number(max*t)}</text></g>)}
      {!probability && <polygon points={band} fill="#e0b98e" opacity=".45"/>}
      <polyline points={line} fill="none" stroke={probability?'#176b61':'#b26529'} strokeWidth="2.5"/>
      {sorted.map((r,i)=><circle key={r.fecha} cx={x(i)} cy={y(r[field])} r="2" fill={probability?'#176b61':'#b26529'}><title>{monthLabel(r.fecha)}: {probability?percent(r[field]):number(r[field],2)}</title></circle>)}
      {[0,Math.floor((sorted.length-1)/2),sorted.length-1].filter((v,i,a)=>a.indexOf(v)===i).map(i=><text key={i} x={x(i)} y={H-8} textAnchor={i===0?'start':i===sorted.length-1?'end':'middle'}>{monthLabel(sorted[i].fecha)}</text>)}
    </svg>
  </figure>;
}
