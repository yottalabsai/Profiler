#!/usr/bin/env node
/**
 * Hook: fires after Write tool completes.
 * If optimizations.json was written, prompt the user to run /backend.
 */

let input = '';
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  try {
    const event = JSON.parse(input);
    const filePath = (event.tool_input && event.tool_input.file_path) || '';
    const isOptimizations = /optimizations\.json$/.test(filePath);

    if (isOptimizations) {
      process.stdout.write(
        '\n[profiler-plugin] optimizations.json written.\n' +
        '  → Run /backend workload.py optimizations.json to generate the custom backend\n'
      );
    }
  } catch (_) {}
  process.exit(0);
});
