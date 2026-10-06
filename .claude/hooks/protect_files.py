"""Guard de herramientas de archivos; no promete aislar Bash ni otros procesos."""

import json
import os
import sys
from pathlib import Path


def main():
    event = json.load(sys.stdin)
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR", event.get("cwd", "."))).resolve()
    tool = event.get("tool_name", "")
    raw_path = event.get("tool_input", {}).get("file_path", "")
    if not raw_path:
        return
    path = Path(raw_path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return
    parts = relative.parts
    reason = None
    if parts[:2] == ("data", "raw"):
        reason = "CSV oficiales protegidos. Ejecutar el pipeline local y leer solo su auditoría agregada; no llevar filas reales al contexto."
    elif tool in ["Write", "Edit"] and parts and parts[0] in ["runs", "delivery", "state"]:
        reason = "Artefactos del harness protegidos contra edición directa. Usar run, promote, bundle o freeze según el flujo."
    elif tool == "Read" and (relative.name in ["oof.csv", "predicciones.csv"] or relative.name.startswith(".env")):
        reason = "No cargar targets OOF ni secretos al contexto de Claude. Usar metrics.json/configs."
    if reason:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}))


if __name__ == "__main__":
    main()
