import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()
lines = [
    "TECTON PNUD. Frente de Claude: datos y modelo.",
    "Leer docs/ARQUITECTURA.md, CLAUDE.md, docs/ESTADO.md y docs/RETO.md.",
    "No editar la estética de dashboard/src: Cursor implementa gráficas y mapas; Laura pule la interfaz.",
    "Hora Bogotá: " + datetime.now(ZoneInfo("America/Bogota")).isoformat(),
]
pointer = root / "state/champion.json"
if pointer.exists():
    champion = json.loads(pointer.read_text())
    lines.append("Champion real conservado: " + champion["run_id"])
    lines.append("Métricas temporales: " + json.dumps(champion.get("metrics", {})))
else:
    lines.append("No hay champion real. Los ensayos sintéticos no validan desempeño competitivo.")
if (root / "state/FROZEN.json").exists():
    lines.append("Entrega congelada. No reemplazar el champion.")
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "\n".join(lines)}}))
