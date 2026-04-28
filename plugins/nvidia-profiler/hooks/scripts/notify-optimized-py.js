#!/usr/bin/env node
/**
 * Hook: fires after Write tool completes.
 * If a *_optimized.py file was written, prompt the user to run /validate.
 */

let input = '';
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  try {
    const event = JSON.parse(input);
    const filePath = (event.tool_input && event.tool_input.file_path) || '';
    const isOptimizedPy = /_optimized\.py$/.test(filePath)
      && !/^test_/.test(require('path').basename(filePath));

    if (isOptimizedPy) {
      const basename = require('path').basename(filePath);
      process.stdout.write(
        `\n[profiler-plugin] ${basename} written.\n` +
        `  → Run /validate ${basename} before profiling\n` +
        `  → Validation catches runtime errors without wasting ncu replay time\n`
      );
    }
  } catch (_) {}
  process.exit(0);
});
