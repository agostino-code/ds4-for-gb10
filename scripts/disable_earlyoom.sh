#!/usr/bin/env bash
# disable_earlyoom.sh - Disable earlyoom and systemd-oomd on DGX Spark
# Prevents premature SIGKILL when loading large models or 256K context in unified memory.

set -euo pipefail

echo "=========================================================="
echo "  NVIDIA DGX Spark - Disabilitazione Servizi OOM Killer"
echo "=========================================================="

if [ "$(id -u)" -ne 0 ]; then
    echo "Questo script richiede privilegi di root. Richiesta sudo in corso..."
    exec sudo "$0" "$@"
fi

# 1. Gestione earlyoom
if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-unit-files earlyoom.service &>/dev/null; then
        echo ">>> Disabilitazione e arresto di 'earlyoom'..."
        systemctl stop earlyoom 2>/dev/null || true
        systemctl disable earlyoom 2>/dev/null || true
        systemctl mask earlyoom 2>/dev/null || true
        echo "    earlyoom: DISABILITATO e MASCHERATO con successo."
    else
        echo "    earlyoom: non presente nel sistema."
    fi

    # 2. Gestione systemd-oomd (Ubuntu 22.04 / 24.04 default)
    if systemctl list-unit-files systemd-oomd.service &>/dev/null; then
        echo ">>> Disabilitazione e arresto di 'systemd-oomd'..."
        systemctl stop systemd-oomd 2>/dev/null || true
        systemctl disable systemd-oomd 2>/dev/null || true
        systemctl mask systemd-oomd 2>/dev/null || true
        echo "    systemd-oomd: DISABILITATO e MASCHERATO con successo."
    else
        echo "    systemd-oomd: non presente nel sistema."
    fi
fi

echo
echo "Verifica stato:"
if pgrep -x earlyoom >/dev/null 2>&1; then
    echo "ATTENZIONE: earlyoom e' ancora attivo con PID $(pgrep -x earlyoom)!"
else
    echo "OK: earlyoom non e' in esecuzione."
fi

if pgrep -x systemd-oomd >/dev/null 2>&1; then
    echo "ATTENZIONE: systemd-oomd e' ancora attivo con PID $(pgrep -x systemd-oomd)!"
else
    echo "OK: systemd-oomd non e' in esecuzione."
fi

echo
echo "Configurazione completata. Ora puoi avviare ./scripts/serve_dgx_spark.sh"
