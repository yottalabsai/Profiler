#!/usr/bin/env node
/**
 * PreToolUse hook: fires before Bash tool executes.
 * Warns if `operator-profiler map` command has --script-args in the wrong position.
 *
 * --script-args uses nargs=argparse.REMAINDER and MUST be the last flag.
 * Any flag placed after --script-args is silently consumed by the workload script,
 * not by `operator-profiler map`, producing empty metrics.raw dicts.
 */

let input = '';
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  try {
    const event = JSON.parse(input);
    const command = (event.tool_input && event.tool_input.command) || '';

    if (!command.includes('operator-profiler map') && !command.includes('operator_profiler map') && !command.includes('nvidia.operator_profiler map')) {
      process.exit(0);
    }

    // Check if any map-level flags appear after --script-args
    const scriptArgsIdx = command.indexOf('--script-args');
    if (scriptArgsIdx === -1) {
      process.exit(0);
    }

    const afterScriptArgs = command.slice(scriptArgsIdx + '--script-args'.length);
    const mapFlags = ['--ncu-sudo', '--ncu-env', '--ncu-path', '--model-name',
                      '--manifest', '--output', '--warmup-iters', '--measure-iters'];
    const misplaced = mapFlags.filter(flag => afterScriptArgs.includes(flag));

    if (misplaced.length > 0) {
      process.stdout.write(
        '\n[profiler-plugin] WARNING: --script-args ordering issue detected!\n' +
        `  Flags found after --script-args: ${misplaced.join(', ')}\n` +
        '  These will be passed to the WORKLOAD SCRIPT, not to operator-profiler map.\n' +
        '  This causes empty metrics.raw in profile.json (silent failure).\n\n' +
        '  CORRECT order:\n' +
        '    operator-profiler map manifest.json \\\n' +
        '        --ncu-sudo true \\\n' +
        '        --ncu-env PYTHONPATH=/repo \\\n' +
        '        --script-args --workload workload.py --compile-backend my_opt\n' +
        '                       ↑ all map flags BEFORE --script-args\n\n'
      );
      // Exit 0 (warning only — do not block the command)
    }
  } catch (_) {}
  process.exit(0);
});
