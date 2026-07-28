#!/usr/bin/env bash
set -e

MAX_ATTEMPTS="${1:-2000}"
DEATH_DIR="death_hunt"
mkdir -p "$DEATH_DIR"

for ((i=1; i<=MAX_ATTEMPTS; i++)); do
    rm -f logs/game.log

    python3 main.py play --my-agent my_agent --n-rounds 1 --no-gui \
        --save-replay --continue-without-training > /dev/null 2>&1

    if grep -q "Agent <my_agent> blown up" logs/game.log 2>/dev/null; then
        echo "Agent died at try #$i"
        cp logs/game.log "$DEATH_DIR/death_${i}_game.log"

        latest_agent_log=$(ls -t logs/my_agent-*.log 2>/dev/null | head -1)
        if [ -n "$latest_agent_log" ]; then
            cp "$latest_agent_log" "$DEATH_DIR/death_${i}_my_agent.log"
        fi

        latest_replay=$(ls -t replays/*.pt 2>/dev/null | head -1)
        if [ -n "$latest_replay" ]; then
            cp "$latest_replay" "$DEATH_DIR/death_${i}_replay.pt"
        fi

        echo "saved as: $DEATH_DIR/death_${i}_*"
        echo "watch as: python3 main.py replay $DEATH_DIR/death_${i}_replay.pt"
        exit 0
    fi

    if (( i % 50 == 0 )); then
        echo "  ... $i rounds survived.."
    fi
done

exit 1