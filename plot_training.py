#!/usr/bin/env python3
"""
Memory-R1 Live Training Dashboard

Reads the JSONL metrics file written during training and produces a
6-panel training dashboard that auto-refreshes.

Usage:
    # On the training server (generates PNG periodically):
    python plot_training.py logs/my_experiment_metrics.jsonl --watch

    # One-shot plot:
    python plot_training.py logs/my_experiment_metrics.jsonl

    # Serve as a tiny HTTP page (great for remote monitoring):
    python plot_training.py logs/my_experiment_metrics.jsonl --serve --port 8888
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_metrics(jsonl_path: str) -> list[dict]:
    """Load metrics from a JSONL file."""
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    # Pad the beginning to keep arrays aligned
    pad = values[:len(values) - len(smoothed)]
    return np.concatenate([pad, smoothed])


def extract_series(records, key, fallback=None):
    """Extract a time series from records."""
    steps = []
    values = []
    for r in records:
        if key in r:
            steps.append(r['step'])
            values.append(r[key])
    if not steps and fallback is not None:
        return [0], [fallback]
    return steps, values


def plot_dashboard(records, output_path, experiment_name="Training"):
    """Generate the 6-panel training dashboard."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{experiment_name} — Live Training Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('#fafafa')

    # Color scheme
    C_ORANGE = '#E67E22'
    C_RED = '#E74C3C'
    C_BLUE = '#3498DB'
    C_GREEN = '#27AE60'
    C_PURPLE = '#9B59B6'
    C_LIGHT = '#BDC3C7'
    SMOOTH_W = 7

    # ---- Panel 1: Reward Score ----
    ax = axes[0, 0]
    steps_mean, vals_mean = extract_series(records, 'critic/score/mean')
    steps_max, vals_max = extract_series(records, 'critic/score/max')
    steps_min, vals_min = extract_series(records, 'critic/score/min')
    # Val score
    steps_val, vals_val = extract_series(records, 'val/test_score/answer_agent')
    if not steps_val:
        # Try other val score keys
        for r in records:
            for k, v in r.items():
                if k.startswith('val/test_score'):
                    steps_val.append(r['step'])
                    vals_val.append(v)
                    break

    if steps_mean:
        ax.fill_between(steps_min, vals_min, vals_max, alpha=0.1, color=C_BLUE)
        ax.plot(steps_mean, vals_mean, color=C_LIGHT, alpha=0.3, linewidth=0.5)
        ax.plot(steps_mean, smooth(np.array(vals_mean), SMOOTH_W),
                color=C_ORANGE, linewidth=2, label='mean (avg)')
        ax.plot(steps_max, smooth(np.array(vals_max), SMOOTH_W),
                color=C_RED, linewidth=1.5, label='max (avg)')
    if steps_val:
        ax.scatter(steps_val, vals_val, color=C_BLUE, marker='*', s=100,
                   zorder=5, label='val score')
    ax.set_title('Reward Score', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Policy Loss ----
    ax = axes[0, 1]
    steps_pg, vals_pg = extract_series(records, 'actor/pg_loss')
    if steps_pg:
        ax.plot(steps_pg, vals_pg, color=C_LIGHT, alpha=0.3, linewidth=0.5)
        ax.plot(steps_pg, smooth(np.array(vals_pg), SMOOTH_W),
                color=C_ORANGE, linewidth=2, label='pg_loss (avg)')
    ax.set_title('Policy Loss', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: KL Divergence ----
    ax = axes[0, 2]
    steps_kl, vals_kl = extract_series(records, 'actor/kl_loss')
    steps_pkk, vals_pkk = extract_series(records, 'critic/kl')
    if steps_kl:
        ax.plot(steps_kl, vals_kl, color=C_LIGHT, alpha=0.3, linewidth=0.5)
        ax.plot(steps_kl, smooth(np.array(vals_kl), SMOOTH_W),
                color=C_ORANGE, linewidth=2, label='kl_loss (avg)')
    if steps_pkk:
        ax.plot(steps_pkk, vals_pkk, color='#90EE90', alpha=0.3, linewidth=0.5)
        ax.plot(steps_pkk, smooth(np.array(vals_pkk), SMOOTH_W),
                color=C_GREEN, linewidth=1.5, label='ppo_kl (avg)')
    ax.set_title('KL Divergence', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('KL')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Gradient Norm ----
    ax = axes[1, 0]
    steps_gn, vals_gn = extract_series(records, 'actor/grad_norm')
    if steps_gn:
        ax.plot(steps_gn, vals_gn, color=C_LIGHT, alpha=0.3, linewidth=0.5)
        ax.plot(steps_gn, smooth(np.array(vals_gn), SMOOTH_W),
                color=C_ORANGE, linewidth=2, label='grad_norm (avg)')
    ax.set_title('Gradient Norm', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 5: Response Length ----
    ax = axes[1, 1]
    steps_rl, vals_rl = extract_series(records, 'response_length/mean')
    if steps_rl:
        ax.plot(steps_rl, vals_rl, color=C_LIGHT, alpha=0.3, linewidth=0.5)
        ax.plot(steps_rl, smooth(np.array(vals_rl), SMOOTH_W),
                color=C_ORANGE, linewidth=2, label='mean (avg)')
    ax.set_title('Response Length', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 6: Clip Fraction ----
    ax = axes[1, 2]
    steps_cf, vals_cf = extract_series(records, 'actor/pg_clipfrac')
    if steps_cf:
        ax.plot(steps_cf, vals_cf, color=C_LIGHT, alpha=0.3, linewidth=0.5)
        ax.plot(steps_cf, smooth(np.array(vals_cf), SMOOTH_W),
                color=C_ORANGE, linewidth=2, label='pg_clipfrac (avg)')
    ax.set_title('Clip Fraction', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def serve_dashboard(png_path, port=8888):
    """Serve the PNG as a simple auto-refreshing HTML page."""
    import http.server
    import threading

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ background: #1a1a2e; margin: 0; display: flex;
               justify-content: center; align-items: center; min-height: 100vh; }}
        img {{ max-width: 98vw; max-height: 96vh; border-radius: 8px;
               box-shadow: 0 4px 20px rgba(0,0,0,0.5); }}
        .info {{ position: fixed; top: 10px; right: 10px; color: #888;
                 font-family: monospace; font-size: 12px; }}
    </style>
</head>
<body>
    <img src="/dashboard.png?t=" + Date.now() alt="Training Dashboard">
    <div class="info">Auto-refreshes every 30s</div>
    <script>
        // Force reload image every 30s without full page refresh
        setInterval(() => {{
            document.querySelector('img').src = '/dashboard.png?t=' + Date.now();
        }}, 30000);
    </script>
</body>
</html>"""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/dashboard.png'):
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                with open(png_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())

        def log_message(self, format, *args):
            pass  # Suppress request logs

    server = http.server.HTTPServer(('0.0.0.0', port), Handler)
    print(f'[Dashboard] Serving at http://0.0.0.0:{port}')
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description='Memory-R1 Live Training Dashboard')
    parser.add_argument('jsonl_path', help='Path to the JSONL metrics file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output PNG path (default: same dir as JSONL)')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Watch mode: regenerate plot every N seconds')
    parser.add_argument('--interval', type=int, default=30,
                        help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--serve', action='store_true',
                        help='Start HTTP server to serve the dashboard')
    parser.add_argument('--port', type=int, default=8888,
                        help='HTTP server port (default: 8888)')
    parser.add_argument('--name', default=None,
                        help='Experiment name for the title')
    args = parser.parse_args()

    if not os.path.exists(args.jsonl_path):
        print(f'[ERROR] File not found: {args.jsonl_path}')
        sys.exit(1)

    output_path = args.output or str(
        Path(args.jsonl_path).with_suffix('.png')
    )
    experiment_name = args.name or Path(args.jsonl_path).stem.replace('_metrics', '')

    def refresh():
        records = load_metrics(args.jsonl_path)
        if not records:
            print('[Dashboard] No metrics yet, waiting...')
            return False
        plot_dashboard(records, output_path, experiment_name)
        print(f'[Dashboard] Updated: {output_path} ({len(records)} steps)')
        return True

    if args.watch or args.serve:
        print(f'[Dashboard] Watching {args.jsonl_path} (refresh every {args.interval}s)')

        if args.serve:
            import threading
            # Generate initial plot
            refresh()
            # Start HTTP server in background
            t = threading.Thread(target=serve_dashboard,
                                 args=(output_path, args.port), daemon=True)
            t.start()

        while True:
            try:
                refresh()
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print('\n[Dashboard] Stopped.')
                break
    else:
        if refresh():
            print(f'[Dashboard] Plot saved to: {output_path}')
        else:
            print('[Dashboard] No metrics found.')
            sys.exit(1)


if __name__ == '__main__':
    main()
