export const COLUMNS = ['DIVIPOLA', 'fecha', 'prob_evento', 'personas_desplazadas_estimadas', 'personas_desplazadas_q10', 'personas_desplazadas_q90'];
export const DEPARTMENTS = {'05':'Antioquia','08':'Atlántico','11':'Bogotá D. C.','13':'Bolívar','15':'Boyacá','17':'Caldas','18':'Caquetá','19':'Cauca','20':'Cesar','23':'Córdoba','25':'Cundinamarca','27':'Chocó','41':'Huila','44':'La Guajira','47':'Magdalena','50':'Meta','52':'Nariño','54':'Norte de Santander','63':'Quindío','66':'Risaralda','68':'Santander','70':'Sucre','73':'Tolima','76':'Valle del Cauca','81':'Arauca','85':'Casanare','86':'Putumayo','88':'San Andrés y Providencia','91':'Amazonas','94':'Guainía','95':'Guaviare','97':'Vaupés','99':'Vichada'};
export const METRICS = {
  prob_evento: {label:'Probabilidad de evento', short:'Probabilidad', unit:'%', scale:'fixed', color:[25, 108, 99]},
  personas_desplazadas_estimadas: {label:'Personas estimadas', short:'Personas', unit:'personas', scale:'log1p', color:[176, 91, 35]},
  amplitud: {label:'Amplitud del intervalo q10–q90', short:'Amplitud', unit:'personas', scale:'log1p', color:[86, 85, 139]},
};
export const number = (v, digits=0) => new Intl.NumberFormat('es-CO', {maximumFractionDigits:digits}).format(v);
export const percent = v => new Intl.NumberFormat('es-CO', {style:'percent',maximumFractionDigits:1}).format(v);
export function monthLabel(value) {
  if (!value) return 'Sin mes';
  return new Intl.DateTimeFormat('es-CO', {month:'short',year:'numeric',timeZone:'UTC'}).format(new Date(`${value.slice(0,7)}-01T12:00:00Z`));
}
const normalizeText = x => String(x).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();

// RFC 4180: separadores y saltos de línea entre comillas; BOM permitido.
export function parseCSV(text) {
  text = text.replace(/^\uFEFF/, '');
  const rows = []; let row = [], field = '', quoted = false, closed = false;
  for (let i=0; i<text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i+1] === '"') {field += '"'; i++;}
      else if (c === '"') {quoted=false; closed=true;}
      else field += c;
    } else if (c === '"') {
      if (field || closed) throw new Error('Comillas inválidas en el CSV.');
      quoted = true;
    } else if (c === ',') {row.push(field); field=''; closed=false;}
    else if (c === '\r' || c === '\n') {
      if (c === '\r' && text[i+1] === '\n') i++;
      row.push(field); if (row.some(x => x !== '')) rows.push(row);
      row=[]; field=''; closed=false;
    } else {
      if (closed) throw new Error('Contenido después de cerrar comillas.');
      field += c;
    }
  }
  if (quoted) throw new Error('Comillas sin cerrar en el CSV.');
  if (field || row.length) {row.push(field); rows.push(row);}
  const header = rows.shift();
  if (!header || header.join(',') !== COLUMNS.join(',')) throw new Error(`Se requieren las seis columnas oficiales, en orden: ${COLUMNS.join(', ')}.`);
  return rows.map((r, i) => {
    if (r.length !== COLUMNS.length) throw new Error(`Fila ${i+2}: número de columnas incorrecto.`);
    return Object.fromEntries(COLUMNS.map((c, j) => [c, r[j]]));
  });
}

export function validateRows(rows) {
  if (!Array.isArray(rows) || !rows.length || rows.length > 100000) throw new Error('Se requieren entre 1 y 100.000 predicciones.');
  const seen = new Set();
  return rows.map((row, i) => {
    const out = {DIVIPOLA:String(row.DIVIPOLA ?? ''), fecha:String(row.fecha ?? '')};
    if (!/^\d{5}$/.test(out.DIVIPOLA)) throw new Error(`Fila ${i+1}: DIVIPOLA debe conservar cinco dígitos.`);
    if (!/^\d{4}-(0[1-9]|1[0-2])-01$/.test(out.fecha)) throw new Error(`Fila ${i+1}: fecha debe ser AAAA-MM-01.`);
    const key = `${out.DIVIPOLA}/${out.fecha}`;
    if (seen.has(key)) throw new Error(`Llave duplicada: ${key}.`);
    seen.add(key);
    for (const c of COLUMNS.slice(2)) {
      const raw = row[c];
      if (raw === null || raw === undefined || typeof raw === 'boolean' || (typeof raw === 'string' && !raw.trim())) throw new Error(`Fila ${i+1}: falta ${c}.`);
      const value = Number(raw);
      if (!Number.isFinite(value) || value < 0 || (c === 'prob_evento' && value > 1)) throw new Error(`Fila ${i+1}: ${c} fuera de dominio.`);
      out[c] = value;
    }
    if (out.personas_desplazadas_q10 > out.personas_desplazadas_q90) throw new Error(`Fila ${i+1}: intervalo cruzado.`);
    out.amplitud = out.personas_desplazadas_q90 - out.personas_desplazadas_q10;
    out.departamento_codigo = out.DIVIPOLA.slice(0,2);
    out.departamento = DEPARTMENTS[out.departamento_codigo] ?? `Departamento ${out.departamento_codigo}`;
    return out;
  });
}

function validateCoordinates(coords) {
  if (!Array.isArray(coords) || !coords.length) throw new Error('Coordenadas geográficas vacías.');
  if (typeof coords[0] === 'number') {
    if (coords.length < 2 || !coords.every(Number.isFinite) || Math.abs(coords[0]) > 180 || Math.abs(coords[1]) > 90) throw new Error('GeoJSON debe estar en longitud/latitud WGS84 (EPSG:4326).');
  } else for (const c of coords) validateCoordinates(c);
}
export function normalizeGeography(geo) {
  if (geo?.type !== 'FeatureCollection' || !Array.isArray(geo.features) || geo.features.length > 3000) throw new Error('Se requiere un GeoJSON FeatureCollection municipal (máximo 3.000 polígonos).');
  if (geo.crs && !/4326|CRS84/.test(JSON.stringify(geo.crs))) throw new Error('Reproyectar la cartografía a EPSG:4326.');
  const seen = new Set();
  const features = geo.features.map(f => {
    const p = Object.fromEntries(Object.entries(f.properties ?? {}).map(([k,v])=>[k.toLowerCase(),v]));
    const code = String(p.divipola ?? p.mpio_cdpmp ?? '');
    if (!/^\d{5}$/.test(code)) throw new Error('Cada polígono necesita DIVIPOLA o mpio_cdpmp de cinco dígitos.');
    if (seen.has(code)) throw new Error(`Geometría duplicada: ${code}. Disolver por DIVIPOLA antes de cargar.`);
    seen.add(code);
    if (!['Polygon','MultiPolygon'].includes(f.geometry?.type)) throw new Error(`Geometría de ${code}: se requiere Polygon o MultiPolygon.`);
    validateCoordinates(f.geometry.coordinates);
    const polygons = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
    for (const polygon of polygons) for (const ring of polygon) {
      if (ring.length < 4 || ring[0][0] !== ring.at(-1)[0] || ring[0][1] !== ring.at(-1)[1]) throw new Error(`Anillo inválido o sin cerrar: ${code}.`);
    }
    return {type:'Feature',geometry:f.geometry,properties:{DIVIPOLA:code,municipio:String(p.municipio ?? p.mpio_cnmbr ?? code),departamento:String(p.departamento ?? p.dpto_cnmbr ?? DEPARTMENTS[code.slice(0,2)] ?? code.slice(0,2))}};
  });
  return {type:'FeatureCollection',features};
}
export function readDataset(text, filename) {
  if (filename.toLowerCase().endsWith('.csv')) return {rows:validateRows(parseCSV(text)),synthetic:null,metadata:{source:filename,run_id:null},geography:null};
  const data = JSON.parse(text);
  if (data.schema_version !== 1) throw new Error('Versión de dashboard JSON incompatible; se requiere schema_version=1.');
  if (data.synthetic !== true && data.synthetic !== false) throw new Error('El JSON debe declarar synthetic=true o false.');
  return {...data, rows:validateRows(data.rows), geography:data.geography ? normalizeGeography(data.geography) : null};
}
export function geographyIndex(geo) {return new Map((geo?.features ?? []).map(f=>[f.properties.DIVIPOLA,f]));}
export function enrichRows(rows, index) {return rows.map(r=>({...r,municipio:index.get(r.DIVIPOLA)?.properties.municipio ?? r.DIVIPOLA}));}
export function filterRows(rows, {month,department='',query=''}) {
  const q = normalizeText(query);
  return rows.filter(r=>(!month || r.fecha === month) && (!department || r.departamento_codigo === department) && (!q || normalizeText(`${r.DIVIPOLA} ${r.municipio} ${r.departamento}`).includes(q)));
}
export function summarize(rows, threshold) {
  return {municipalities:rows.length,above:rows.filter(r=>r.prob_evento >= threshold).length,people:rows.reduce((s,r)=>s+r.personas_desplazadas_estimadas,0)};
}
export function selectGeoJSON(rows,index) {
  return {type:'FeatureCollection',features:rows.filter(r=>index.has(r.DIVIPOLA)).map(r=>({...index.get(r.DIVIPOLA),properties:{...r,log_personas:Math.log1p(r.personas_desplazadas_estimadas),log_amplitud:Math.log1p(r.amplitud)}}))};
}
export function toCSV(rows) {
  return COLUMNS.join(',')+'\n'+rows.map(r=>COLUMNS.map(c=>r[c]).join(',')).join('\n')+'\n';
}
