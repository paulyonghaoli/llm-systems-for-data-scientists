/* Web Worker: runs learner Python via Pyodide, off the main thread.
 * Protocol: in  {id, setup, code, tests}
 *           out {id, type: "progress", msg} | {id, type: "done", result}
 * result: {status: "ok"|"pass"|"fail"|"error", stdout, error}
 */
/* global loadPyodide, importScripts */
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

var RUNNER_SRC = [
  "import io, json, sys, traceback",
  "def _llm_run(setup, code, tests):",
  "    ns = {}",
  "    buf = io.StringIO()",
  "    old = sys.stdout",
  "    sys.stdout = buf",
  "    try:",
  "        try:",
  "            exec(setup, ns)",
  "        except Exception:",
  "            return json.dumps({'status': 'error', 'stdout': buf.getvalue(),",
  "                               'error': 'Exercise setup failed: ' + traceback.format_exc(limit=1)})",
  "        try:",
  "            exec(code, ns)",
  "        except Exception:",
  "            tb = traceback.format_exc(limit=2)",
  "            return json.dumps({'status': 'error', 'stdout': buf.getvalue(), 'error': tb})",
  "        if tests.strip():",
  "            try:",
  "                exec(tests, ns)",
  "            except AssertionError as e:",
  "                return json.dumps({'status': 'fail', 'stdout': buf.getvalue(),",
  "                                   'error': str(e) or 'an assertion failed'})",
  "            except Exception:",
  "                tb = traceback.format_exc(limit=2)",
  "                return json.dumps({'status': 'error', 'stdout': buf.getvalue(), 'error': tb})",
  "            return json.dumps({'status': 'pass', 'stdout': buf.getvalue(), 'error': ''})",
  "        return json.dumps({'status': 'ok', 'stdout': buf.getvalue(), 'error': ''})",
  "    finally:",
  "        sys.stdout = old",
].join("\n");

var pyodidePromise = null;

function getPyodide(progress) {
  if (!pyodidePromise) {
    pyodidePromise = (async function () {
      progress("Loading Python runtime (~10 MB, cached after first load)…");
      var py = await loadPyodide();
      progress("Loading NumPy…");
      await py.loadPackage("numpy");
      py.runPython(RUNNER_SRC);
      return py;
    })();
  }
  return pyodidePromise;
}

self.onmessage = async function (e) {
  var d = e.data;
  function progress(msg) { self.postMessage({ id: d.id, type: "progress", msg: msg }); }
  try {
    var py = await getPyodide(progress);
    progress("Running…");
    var runner = py.globals.get("_llm_run");
    var resultJson = runner(d.setup, d.code, d.tests);
    if (runner.destroy) runner.destroy();
    self.postMessage({ id: d.id, type: "done", result: JSON.parse(resultJson) });
  } catch (err) {
    self.postMessage({
      id: d.id, type: "done",
      result: { status: "error", stdout: "", error: "Runtime error: " + String(err) },
    });
  }
};
