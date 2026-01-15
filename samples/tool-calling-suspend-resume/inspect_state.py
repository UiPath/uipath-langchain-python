"""Inspect the contents of state.db checkpoints."""

import sqlite3
import msgpack
import json
from pathlib import Path

db_path = Path("__uipath/state.db")

def decode_msgpack(data):
    """Decode msgpack binary data."""
    if not data:
        return None
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as e:
        return f"<decode error: {e}>"

def inspect_checkpoints():
    """Show checkpoint data."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    print("=" * 80)
    print("CHECKPOINTS")
    print("=" * 80)

    cursor.execute("""
        SELECT thread_id, checkpoint_id, parent_checkpoint_id,
               checkpoint, metadata
        FROM checkpoints
        ORDER BY rowid
    """)

    for idx, (thread_id, checkpoint_id, parent_id, checkpoint_blob, metadata_blob) in enumerate(cursor.fetchall(), 1):
        print(f"\n{'='*80}")
        print(f"Checkpoint #{idx}")
        print(f"{'='*80}")
        print(f"Thread ID: {thread_id}")
        print(f"Checkpoint ID: {checkpoint_id}")
        print(f"Parent ID: {parent_id or 'None (root)'}")

        # Decode checkpoint
        checkpoint_data = decode_msgpack(checkpoint_blob)
        print(f"\nCheckpoint Data:")
        if isinstance(checkpoint_data, dict):
            # Pretty print the checkpoint structure
            for key, value in checkpoint_data.items():
                if key == 'channel_values':
                    print(f"  {key}:")
                    for channel, channel_value in value.items():
                        print(f"    {channel}: {channel_value}")
                elif key == 'versions_seen':
                    print(f"  {key}: {value}")
                elif key == 'pending_sends':
                    print(f"  {key}: {len(value)} items" if value else "  pending_sends: []")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {checkpoint_data}")

        # Decode metadata
        metadata = decode_msgpack(metadata_blob)
        print(f"\nMetadata: {metadata}")

    print("\n" + "=" * 80)
    print("WRITES (Interrupt & Resume Data)")
    print("=" * 80)

    cursor.execute("""
        SELECT checkpoint_id, task_id, idx, channel, value
        FROM writes
        WHERE channel IN ('__interrupt__', '__resume__', 'query', 'result')
        ORDER BY checkpoint_id, idx
    """)

    for checkpoint_id, task_id, idx, channel, value_blob in cursor.fetchall():
        value = decode_msgpack(value_blob)
        print(f"\nCheckpoint: {checkpoint_id[:20]}...")
        print(f"  Channel: {channel}")
        print(f"  Task ID: {task_id[:20]}...")
        print(f"  Index: {idx}")
        print(f"  Value: {value}")

    conn.close()

if __name__ == "__main__":
    if not db_path.exists():
        print(f"❌ No state.db found at {db_path}")
    else:
        inspect_checkpoints()
