#!/usr/bin/env node
/**
 * Hook: fires after Write tool completes.
 * If profile.json was written, prompt the user to run /analyze.
 */

let input = '';
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  try {
    const event = JSON.parse(input);
    const filePath = (event.tool_input && event.tool_input.file_path) || '';
    const isProfile = /profile\.json$/.test(filePath)
      && !filePath.includes('profile_optimized');

    if (isProfile) {
      process.stdout.write(
        '\n[profiler-plugin] profile.json written.\n' +
        '  → Run /analyze to triage bottlenecks\n' +
        '  → Or run /optimize workload.py for the full end-to-end workflow\n'
      );
    }
  } catch (_) {
    // Non-JSON input or missing fields — silently exit
  }
  process.exit(0);
});
