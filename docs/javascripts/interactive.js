/* LLM Systems for Data Scientists — interactive components (vanilla JS, no build step).
 *
 * <quiz-bank src="BANK_ID">      — interactive MCQ/numeric quiz, YAML-authored,
 *                                  converted to JSON at mkdocs build time.
 * <code-exercise src="EX_ID">    — DataCamp-style in-browser Python exercise
 *                                  (CodeMirror editor + Pyodide in a Web Worker).
 *
 * Progress persists in localStorage under "llmds.*" keys. No backend.
 */
(function () {
  "use strict";

  // Site base URL, derived from this script's own <script src>.
  var SITE_BASE = (function () {
    var s = document.currentScript;
    if (s && s.src) return s.src.replace(/javascripts\/interactive\.js.*$/, "");
    return "/";
  })();

  function fetchJSON(rel) {
    return fetch(SITE_BASE + rel).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status + " for " + rel);
      return r.json();
    });
  }

  function store(key, val) {
    try { localStorage.setItem("llmds." + key, JSON.stringify(val)); } catch (e) { /* private mode */ }
  }
  function load(key) {
    try {
      var v = localStorage.getItem("llmds." + key);
      return v ? JSON.parse(v) : null;
    } catch (e) { return null; }
  }

  // Minimal markdown: escapes HTML, then renders `code`, **bold**, newlines.
  // MathJax spans pass through untouched and are typeset afterwards.
  //
  // Code spans are lifted out before the bold pass and put back afterwards.
  // Markdown treats a code span's contents as literal and running the bold
  // regex over the whole string does not: an exponent inside backticks, like
  // `f**K`, had its ** paired with a real ** later in the text, so the bold
  // ran from inside the <code> element out into the surrounding prose and
  // swallowed everything between. Substituting first and scanning second is
  // the bug; lifting out first is what a real renderer does.
  var CODE_MARK = String.fromCharCode(0xE000);

  function mdLite(text) {
    var h = String(text)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // A private-use character cannot appear in the escaped text above, which
    // is what makes it safe as a placeholder delimiter.
    var spans = [];
    h = h.replace(/`([^`]+)`/g, function (_, code) {
      spans.push(code);
      return CODE_MARK + (spans.length - 1) + CODE_MARK;
    });

    h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    var restore = new RegExp(CODE_MARK + "(\\d+)" + CODE_MARK, "g");
    h = h.replace(restore, function (_, i) {
      return "<code>" + spans[Number(i)] + "</code>";
    });

    return h.replace(/\n/g, "<br>");
  }

  function typeset(el) {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([el]).catch(function () {});
    }
  }

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }

  /* ------------------------------------------------------------------ *
   *  Quiz bank
   * ------------------------------------------------------------------ */

  var QuizBank = /** @class */ (function () {
    function attach(host) {
      var id = host.getAttribute("src");
      fetchJSON("assets/generated/quizzes/" + id + ".json")
        .then(function (bank) { render(host, id, bank); })
        .catch(function (e) {
          host.appendChild(el("div", "llm-card", "Quiz bank <code>" + id + "</code> failed to load: " + e.message));
        });
    }

    function render(host, bankId, bank) {
      var card = el("div", "llm-card");
      var header = el("div", "llm-bank-header");
      header.appendChild(el("span", "", bank.title || "Check your understanding"));
      var score = el("span", "llm-score", "");
      header.appendChild(score);
      card.appendChild(header);

      var state = load("quiz." + bankId) || {};

      function updateScore() {
        var correct = 0;
        bank.questions.forEach(function (q) {
          if (state[q.id] && state[q.id].correct) correct++;
        });
        score.textContent = correct + " / " + bank.questions.length + " correct";
        store("quiz." + bankId, state);
      }

      bank.questions.forEach(function (q, idx) {
        card.appendChild(renderQuestion(q, idx + 1, state, updateScore));
      });

      updateScore();
      host.appendChild(card);
      typeset(card);
    }

    function renderQuestion(q, num, state, onChange) {
      var wrap = el("div", "llm-q");
      var prompt = el("div", "llm-q-prompt");
      prompt.innerHTML =
        '<span class="llm-q-num">' + num + ".</span>" + mdLite(q.prompt) +
        (q.difficulty ? '<span class="llm-q-tag">' + q.difficulty + "</span>" : "");
      wrap.appendChild(prompt);

      var inputName = "llm-" + q.id + "-" + Math.random().toString(36).slice(2, 7);
      var body = el("div");
      wrap.appendChild(body);

      var feedback = el("div", "llm-feedback");
      var checkBtn = el("button", "llm-btn", "Check");
      var retryBtn = el("button", "llm-btn llm-btn-ghost", "Retry");
      retryBtn.style.display = "none";

      function buildOptions() {
        body.innerHTML = "";
        feedback.textContent = "";
        feedback.className = "llm-feedback";
        if (q.type === "numeric") {
          var input = el("input", "llm-numeric-input");
          input.type = "text";
          input.placeholder = q.placeholder || "numeric answer";
          body.appendChild(input);
        } else {
          var kind = q.type === "multi" ? "checkbox" : "radio";
          q.options.forEach(function (opt, i) {
            var label = el("label", "llm-opt");
            var inp = document.createElement("input");
            inp.type = kind;
            inp.name = inputName;
            inp.value = String(i);
            label.appendChild(inp);
            label.appendChild(el("span", "", mdLite(opt.text)));
            body.appendChild(label);
          });
        }
      }

      function check() {
        var correct = false;
        if (q.type === "numeric") {
          var input = body.querySelector("input");
          var v = parseFloat(input.value.replace(",", "."));
          var tol = q.tolerance !== undefined ? q.tolerance : 1e-3;
          correct = isFinite(v) && Math.abs(v - q.answer) <= tol;
          feedback.textContent = correct
            ? "Correct." : "Not quite" + (q.hint ? " — " + q.hint : ".");
          if (correct && q.explanation) feedback.textContent += " " + q.explanation;
        } else {
          var chosen = [].slice.call(body.querySelectorAll("input:checked"))
            .map(function (i) { return parseInt(i.value, 10); });
          if (!chosen.length) { feedback.textContent = "Pick an answer first."; return; }
          var right = q.options
            .map(function (o, i) { return o.correct ? i : -1; })
            .filter(function (i) { return i >= 0; });
          correct = chosen.length === right.length &&
            chosen.every(function (c) { return right.indexOf(c) >= 0; });

          [].slice.call(body.querySelectorAll(".llm-opt")).forEach(function (label, i) {
            var opt = q.options[i];
            var picked = chosen.indexOf(i) >= 0;
            label.querySelector("input").disabled = true;
            if (opt.correct) label.classList.add("llm-correct");
            else if (picked) label.classList.add("llm-wrong");
            if ((picked || opt.correct) && opt.explanation) {
              var ex = el("div", "llm-explain", mdLite(opt.explanation));
              label.insertAdjacentElement("afterend", ex);
            }
          });
          feedback.textContent = correct ? "Correct." : "Not quite — see the notes above.";
        }
        feedback.classList.add(correct ? "ok" : "bad");
        state[q.id] = { correct: correct, answered: true };
        onChange();
        checkBtn.style.display = "none";
        retryBtn.style.display = "";
        typeset(wrap);
      }

      checkBtn.addEventListener("click", check);
      retryBtn.addEventListener("click", function () {
        buildOptions();
        checkBtn.style.display = "";
        retryBtn.style.display = "none";
      });

      buildOptions();
      wrap.appendChild(checkBtn);
      wrap.appendChild(retryBtn);
      wrap.appendChild(feedback);
      return wrap;
    }

    return { attach: attach };
  })();

  /* ------------------------------------------------------------------ *
   *  Python runtime (Pyodide in a Web Worker), shared across exercises
   * ------------------------------------------------------------------ */

  var Py = {
    worker: null,
    seq: 0,
    pending: {},
    ensure: function () {
      if (!this.worker) {
        var self_ = this;
        this.worker = new Worker(SITE_BASE + "javascripts/py-worker.js");
        this.worker.onmessage = function (e) {
          var d = e.data;
          var p = self_.pending[d.id];
          if (!p) return;
          if (d.type === "progress") { if (p.onProgress) p.onProgress(d.msg); return; }
          delete self_.pending[d.id];
          p.resolve(d.result);
        };
        this.worker.onerror = function (err) {
          Object.keys(self_.pending).forEach(function (id) {
            self_.pending[id].resolve({
              status: "error", stdout: "",
              error: "Python runtime failed to load (network required for first run): " + err.message,
            });
            delete self_.pending[id];
          });
        };
      }
      return this.worker;
    },
    run: function (payload, onProgress) {
      var self_ = this;
      return new Promise(function (resolve) {
        var id = ++self_.seq;
        self_.pending[id] = { resolve: resolve, onProgress: onProgress };
        self_.ensure().postMessage({
          id: id, setup: payload.setup || "", code: payload.code || "", tests: payload.tests || "",
        });
      });
    },
  };

  /* ------------------------------------------------------------------ *
   *  Code exercise
   * ------------------------------------------------------------------ */

    /* The learner never sees setup_code, so a name alone leaves them guessing
     * at signatures. Generated from the real objects at build time. */
    function providedPanel(items) {
      var d = document.createElement("details");
      d.className = "llm-provided";
      var s = document.createElement("summary");
      var nfn = items.filter(function (i) { return i.kind !== "constant"; }).length;
      s.textContent = "Provided in this exercise — " + items.length + " object" +
        (items.length === 1 ? "" : "s") + (nfn ? " (" + nfn + " callable)" : "");
      d.appendChild(s);
      var dl = el("div", "llm-provided__list");
      items.forEach(function (it) {
        var row = el("div", "llm-provided__item");
        var sig = el("code", "llm-provided__sig");
        sig.textContent = it.signature || (it.name + " = " + it.value);
        row.appendChild(sig);
        if (it.summary) row.appendChild(el("div", "llm-provided__doc", mdLite(it.summary)));
        // A signature names the parameters; it does not say what they mean
        // or what units they are in, and the source is hidden.
        if ((it.params || []).length || it.returns) {
          var tbl = document.createElement("table");
          tbl.className = "llm-provided__params";
          var addRow = function (label, type, doc, cls) {
            var tr = tbl.insertRow();
            if (cls) tr.className = cls;
            var c0 = tr.insertCell();
            var code = document.createElement("code");
            code.textContent = label;
            c0.appendChild(code);
            var c1 = tr.insertCell();
            if (type) {
              var em = document.createElement("em");
              em.textContent = type;
              c1.appendChild(em);
            }
            tr.insertCell().innerHTML = mdLite(doc);
          };
          (it.params || []).forEach(function (pm) {
            addRow(pm.name, pm.type, pm.doc, "");
          });
          if (it.returns) addRow("returns", it.returns.type, it.returns.doc, "is-return");
          row.appendChild(tbl);
        }
        if ((it.notes || []).length) {
          var ul = document.createElement("ul");
          ul.className = "llm-provided__notes";
          it.notes.forEach(function (n) {
            var li = document.createElement("li");
            li.innerHTML = mdLite(n);
            ul.appendChild(li);
          });
          row.appendChild(ul);
        }
        if (it.example) {
          var ex = el("pre", "llm-provided__eg");
          // Output is computed at build time by actually running the call, so
          // a worked example here can never drift from the code.
          ex.textContent = ">>> " + it.example +
            (it.example_out !== undefined ? "\n" + it.example_out : "");
          row.appendChild(ex);
        }
        dl.appendChild(row);
      });
      d.appendChild(dl);
      return d;
    }

  var CodeExercise = /** @class */ (function () {
    function attach(host) {
      var id = host.getAttribute("src");
      fetchJSON("assets/generated/exercises/" + id + ".json")
        .then(function (spec) { render(host, id, spec); })
        .catch(function (e) {
          host.appendChild(el("div", "llm-card", "Exercise <code>" + id + "</code> failed to load: " + e.message));
        });
    }

    function render(host, exId, spec) {
      var card = el("div", "llm-card");
      var header = el("div", "llm-bank-header");
      header.appendChild(el("span", "llm-ex-title", "🧪 " + spec.title));
      var status = el("span", "llm-status", "not attempted");
      header.appendChild(status);
      card.appendChild(header);
      if (spec.description) card.appendChild(el("div", "llm-ex-desc", mdLite(spec.description)));
      if ((spec.provided || []).length) card.appendChild(providedPanel(spec.provided));

      var saved = load("ex." + exId) || {};
      if (saved.passed) setStatus("pass");

      var wrap = el("div", "llm-editor-wrap");
      var ta = document.createElement("textarea");
      ta.className = "llm-plain";
      ta.value = saved.code || spec.starter_code || "";
      ta.spellcheck = false;
      wrap.appendChild(ta);
      card.appendChild(wrap);

      var cm = null;
      if (window.CodeMirror) {
        cm = window.CodeMirror.fromTextArea(ta, {
          mode: "python", lineNumbers: true, indentUnit: 4, viewportMargin: Infinity,
        });
        cm.on("change", persistSoon);
      } else {
        ta.addEventListener("input", persistSoon);
      }
      function getCode() { return cm ? cm.getValue() : ta.value; }
      function setCode(v) { if (cm) cm.setValue(v); else ta.value = v; }

      var persistTimer = null;
      function persistSoon() {
        clearTimeout(persistTimer);
        persistTimer = setTimeout(function () {
          saved.code = getCode();
          store("ex." + exId, saved);
        }, 500);
      }

      function setStatus(kind) {
        status.className = "llm-status" + (kind ? " " + kind : "");
        status.textContent = kind === "pass" ? "passed ✓" : kind === "fail" ? "not passing" : "not attempted";
      }

      var output = el("div", "llm-output");
      var hintBox = el("div");
      var solBox = el("div", "llm-solution");
      var hintIdx = 0;

      var runBtn = el("button", "llm-btn", "▶ Run");
      var submitBtn = el("button", "llm-btn", "✓ Submit");
      var hintBtn = el("button", "llm-btn llm-btn-ghost", "Hint");
      var resetBtn = el("button", "llm-btn llm-btn-ghost", "Reset");
      var solBtn = el("button", "llm-btn llm-btn-ghost", "Show solution");

      function execute(withTests) {
        output.textContent = "";
        runBtn.disabled = submitBtn.disabled = true;
        var progressLine = el("div", "", "Starting…");
        output.appendChild(progressLine);
        Py.run(
          { setup: spec.setup_code, code: getCode(), tests: withTests ? spec.tests : "" },
          function (msg) { progressLine.textContent = msg; }
        ).then(function (res) {
          runBtn.disabled = submitBtn.disabled = false;
          output.textContent = "";
          if (res.stdout) output.appendChild(el("div", "", "").appendChild(document.createTextNode(res.stdout)).parentNode);
          if (res.status === "pass") {
            output.appendChild(el("div", "okline", "✓ All checks passed. Nicely done."));
            saved.passed = true;
            store("ex." + exId, saved);
            setStatus("pass");
          } else if (res.status === "fail") {
            output.appendChild(el("div", "err", "✗ Check failed: " + res.error));
            setStatus("fail");
          } else if (res.status === "error") {
            output.appendChild(el("div", "err", res.error));
            if (withTests) setStatus("fail");
          } else if (!res.stdout) {
            output.appendChild(el("div", "", "(no output — use print() to inspect values)"));
          }
        });
      }

      runBtn.addEventListener("click", function () { execute(false); });
      submitBtn.addEventListener("click", function () { execute(true); });
      hintBtn.addEventListener("click", function () {
        var hints = spec.hints || [];
        if (!hints.length) return;
        if (hintIdx < hints.length) {
          hintBox.appendChild(el("div", "llm-hint", "Hint " + (hintIdx + 1) + "/" + hints.length + ": " + mdLite(hints[hintIdx])));
          hintIdx++;
          typeset(hintBox);
        }
        if (hintIdx >= hints.length) hintBtn.disabled = true;
      });
      resetBtn.addEventListener("click", function () {
        if (confirm("Reset this exercise to the starter code?")) {
          setCode(spec.starter_code || "");
          persistSoon();
        }
      });
      solBtn.addEventListener("click", function () {
        if (solBox.childNodes.length) { solBox.innerHTML = ""; solBtn.textContent = "Show solution"; return; }
        if (!saved.passed && !confirm("Give it a real try first — show the solution anyway?")) return;
        var pre = el("pre");
        pre.textContent = spec.solution || "(no solution provided)";
        solBox.appendChild(pre);
        solBtn.textContent = "Hide solution";
      });

      card.appendChild(runBtn);
      card.appendChild(submitBtn);
      if ((spec.hints || []).length) card.appendChild(hintBtn);
      card.appendChild(resetBtn);
      if (spec.solution) card.appendChild(solBtn);
      card.appendChild(output);
      card.appendChild(hintBox);
      card.appendChild(solBox);
      host.appendChild(card);
      typeset(card);
      if (cm) setTimeout(function () { cm.refresh(); }, 50);
    }

    return { attach: attach };
  })();

  /* ------------------------------------------------------------------ */

  function init() {
    [].slice.call(document.querySelectorAll("quiz-bank")).forEach(QuizBank.attach);
    [].slice.call(document.querySelectorAll("code-exercise")).forEach(CodeExercise.attach);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
