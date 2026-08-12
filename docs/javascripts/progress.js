/* Per-lesson progress marking.
 *
 * Stored in localStorage, not cookies. Cookies are sent to the server on
 * every request, which is pure waste for a static site and is the thing
 * that makes consent banners necessary. Nothing here is transmitted
 * anywhere — it never leaves the browser — so there is nothing to consent
 * to and nothing to ask about.
 *
 * The cost of that choice is that progress is per-browser and is lost if
 * site data is cleared, so the summary page offers export/import.
 */
(function () {
  "use strict";

  var KEY = "llmds:progress:v1";

  // Same derivation interactive.js uses: take the base from this script's
  // own src rather than assuming the site sits at the domain root.
  var SITE_BASE = (function () {
    var s = document.currentScript;
    if (s && s.src) return s.src.replace(/javascripts\/progress\.js.*$/, "");
    return "/";
  })();

  // SITE_BASE is a full URL; the stored keys are pathnames, so keep a
  // pathname-only form for comparing against manifest entries.
  var BASE_PATH = (function () {
    try {
      return new URL(SITE_BASE, window.location.href).pathname;
    } catch (e) {
      return "/";
    }
  })();

  var STATES = [
    { id: "reading", label: "Reading", icon: "◐", hint: "Started this one" },
    { id: "done", label: "Done", icon: "✓", hint: "Finished and understood" },
    { id: "revisit", label: "Needs more time", icon: "↻",
      hint: "Too hard for now — come back to it" }
  ];

  function load() {
    try {
      return JSON.parse(localStorage.getItem(KEY)) || {};
    } catch (e) {
      return {};
    }
  }

  function save(data) {
    try {
      localStorage.setItem(KEY, JSON.stringify(data));
      return true;
    } catch (e) {
      return false;   // private mode, or quota — fail quietly, never break the page
    }
  }

  /* Normalize so /a/b/ and /a/b/index.html agree. */
  function pageKey(path) {
    var p = path || window.location.pathname;
    p = p.replace(/index\.html$/, "").replace(/\.html$/, "/");
    if (p.charAt(p.length - 1) !== "/") p += "/";
    return p;
  }

  function get(key) {
    var rec = load()[key || pageKey()];
    return rec ? rec.state : null;
  }

  function set(state) {
    var data = load();
    var k = pageKey();
    if (state === null) {
      delete data[k];
    } else {
      data[k] = { state: state, at: new Date().toISOString(), title: docTitle() };
    }
    save(data);
    document.dispatchEvent(new CustomEvent("llmds:progress-changed"));
  }

  function docTitle() {
    var h1 = document.querySelector("article h1");
    if (!h1) return document.title;
    // Material appends a permalink anchor inside the heading; its pilcrow
    // is part of textContent and would otherwise be stored in the title.
    var clone = h1.cloneNode(true);
    clone.querySelectorAll(".headerlink").forEach(function (a) { a.remove(); });
    return clone.textContent.replace(/¶/g, "").trim();
  }

  /* ---- the per-page control ---------------------------------------- */

  function buildControl() {
    var wrap = document.createElement("div");
    wrap.className = "llmp-progress";
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", "Mark your progress on this page");

    var label = document.createElement("span");
    label.className = "llmp-progress__label";
    label.textContent = "Mark this page:";
    wrap.appendChild(label);

    STATES.forEach(function (s) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "llmp-progress__btn";
      b.dataset.state = s.id;
      b.title = s.hint;
      b.innerHTML = '<span aria-hidden="true">' + s.icon + "</span> " + s.label;
      b.addEventListener("click", function () {
        // Clicking the active state clears it, so a mis-click is undoable
        // without a separate "clear" affordance.
        set(get() === s.id ? null : s.id);
        paint(wrap);
      });
      wrap.appendChild(b);
    });

    var note = document.createElement("span");
    note.className = "llmp-progress__note";
    note.textContent = "saved in this browser only";
    wrap.appendChild(note);

    paint(wrap);
    return wrap;
  }

  function paint(wrap) {
    var cur = get();
    wrap.querySelectorAll(".llmp-progress__btn").forEach(function (b) {
      var on = b.dataset.state === cur;
      b.setAttribute("aria-pressed", on ? "true" : "false");
      b.classList.toggle("is-on", on);
    });
    wrap.dataset.state = cur || "";
  }

  /* Only on study pages: module lessons and the graded project briefs. */
  function isStudyPage() {
    return /\/modules\//.test(window.location.pathname);
  }

  function injectControl() {
    if (!isStudyPage()) return;
    var article = document.querySelector("article.md-content__inner, article");
    if (!article || article.querySelector(".llmp-progress")) return;
    var h1 = article.querySelector("h1");
    if (!h1) return;
    // After the metadata line and its rule, so the page still opens with
    // its own title and status badge.
    var anchor = h1.nextElementSibling;
    while (anchor && anchor.tagName !== "HR") anchor = anchor.nextElementSibling;
    var control = buildControl();
    if (anchor && anchor.parentNode) {
      anchor.parentNode.insertBefore(control, anchor.nextSibling);
    } else {
      h1.parentNode.insertBefore(control, h1.nextSibling);
    }
  }

  /* ---- the summary ------------------------------------------------- */

  function pct(n, d) { return d ? Math.round((100 * n) / d) : 0; }

  function renderSummary(host, lessons) {
    var data = load();
    var counts = { done: 0, reading: 0, revisit: 0 };
    Object.keys(data).forEach(function (k) {
      if (counts[data[k].state] !== undefined) counts[data[k].state]++;
    });
    var total = lessons.length || 0;

    var html = "";
    html += '<div class="llmp-sum__bar" role="img" aria-label="' +
      counts.done + " of " + total + ' lessons done">';
    html += '<span class="is-done" style="width:' + pct(counts.done, total) + '%"></span>';
    html += '<span class="is-reading" style="width:' + pct(counts.reading, total) + '%"></span>';
    html += '<span class="is-revisit" style="width:' + pct(counts.revisit, total) + '%"></span>';
    html += "</div>";
    html += '<p class="llmp-sum__counts"><strong>' + counts.done + "</strong> done · <strong>" +
      counts.reading + "</strong> reading · <strong>" + counts.revisit +
      "</strong> needs more time · " + total + " pages total</p>";

    if (counts.revisit) {
      html += "<h3>Come back to these</h3><ul>";
      Object.keys(data).forEach(function (k) {
        if (data[k].state === "revisit") {
          html += '<li><a href="' + k + '">' + escapeHtml(data[k].title || k) + "</a></li>";
        }
      });
      html += "</ul>";
    }

    if (total) {
      var byModule = {};
      lessons.forEach(function (l) {
        // Key on the directory slug so the rows stay in curriculum order,
        // and display the module's own title.
        var k = l.slug || l.module;
        (byModule[k] = byModule[k] || []).push(l);
      });
      html += "<h3>By module</h3><table><thead><tr><th>Module</th><th>Done</th></tr></thead><tbody>";
      Object.keys(byModule).sort().forEach(function (k) {
        var ls = byModule[k];
        var m = ls[0].module || k;
        var d = ls.filter(function (l) {
          var r = data[pageKey(BASE_PATH + l.url)];
          return r && r.state === "done";
        }).length;
        html += "<tr><td>" + escapeHtml(m) + "</td><td>" + d + " / " + ls.length + "</td></tr>";
      });
      html += "</tbody></table>";
    }

    html += '<p class="llmp-sum__io">' +
      '<button type="button" class="llmp-progress__btn" data-act="export">Export JSON</button> ' +
      '<button type="button" class="llmp-progress__btn" data-act="import">Import JSON</button> ' +
      '<button type="button" class="llmp-progress__btn" data-act="reset">Clear all</button></p>';

    host.innerHTML = html;
    wireIo(host, lessons);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function wireIo(host, lessons) {
    host.querySelectorAll("[data-act]").forEach(function (b) {
      b.addEventListener("click", function () {
        var act = b.dataset.act;
        if (act === "export") {
          var blob = new Blob([JSON.stringify(load(), null, 2)], { type: "application/json" });
          var a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = "robotics-progress.json";
          a.click();
          URL.revokeObjectURL(a.href);
        } else if (act === "import") {
          var inp = document.createElement("input");
          inp.type = "file";
          inp.accept = "application/json,.json";
          inp.addEventListener("change", function () {
            var f = inp.files && inp.files[0];
            if (!f) return;
            var fr = new FileReader();
            fr.onload = function () {
              try {
                var incoming = JSON.parse(fr.result);
                var merged = load();
                // Newest mark per page wins, so importing an older export
                // cannot silently undo work done since.
                Object.keys(incoming).forEach(function (k) {
                  var a2 = merged[k], b2 = incoming[k];
                  if (!a2 || (b2 && b2.at > a2.at)) merged[k] = b2;
                });
                save(merged);
                renderSummary(host, lessons);
              } catch (e) {
                window.alert("That file could not be read as progress JSON.");
              }
            };
            fr.readAsText(f);
          });
          inp.click();
        } else if (act === "reset") {
          if (window.confirm("Clear all progress marks in this browser?")) {
            save({});
            renderSummary(host, lessons);
          }
        }
      });
    });
  }

  function initSummary() {
    var host = document.querySelector("progress-summary");
    if (!host) return;
    fetch(SITE_BASE + "assets/generated/lessons.json")
      .then(function (r) { return r.ok ? r.json() : []; })
      .catch(function () { return []; })
      .then(function (lessons) { renderSummary(host, lessons || []); });
  }

  function init() {
    injectControl();
    initSummary();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  // Material ships an instant-loading router; re-run on navigation.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  }
})();
