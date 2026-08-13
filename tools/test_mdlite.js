/*
 * Gate 29: the widget's markdown renderer must protect code spans.
 *
 * `mdLite` in docs/javascripts/interactive.js renders exercise descriptions
 * and quiz explanations. It once substituted <code> elements first and then
 * ran the bold regex over the whole string, so a `**` inside a code span --
 * an exponent, like `f**K` -- paired with a real `**` later in the text and
 * bolded everything between them, breaking across the <code> boundary.
 *
 * No other gate could see this. The source YAML is correct markdown, the word
 * counts are unchanged, the components resolve, and the page renders without
 * an error; only the visible output is wrong. So the check is on the renderer
 * itself, exercised the way the browser exercises it.
 *
 *     node tools/test_mdlite.js
 */

"use strict";

const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "..", "docs", "javascripts", "interactive.js");
const source = fs.readFileSync(SRC, "utf8");

// Parse the whole file first: a syntax error here breaks every widget on the
// site, and mkdocs will happily publish it.
try {
  new Function(source);
} catch (e) {
  console.error("interactive.js does not parse: " + e.message);
  process.exit(1);
}

const start = source.indexOf("var CODE_MARK");
const end = source.indexOf("function typeset");
if (start === -1 || end === -1 || end < start) {
  console.error("could not locate mdLite in interactive.js — has it been renamed?");
  process.exit(1);
}
const mdLite = new Function(source.slice(start, end) + "; return mdLite;")();

const cases = [
  {
    name: "exponent in a code span does not pair with later bold",
    input: "cost is `(1 - f**K) / (1 - f)` and the **success rate** is `1 - f**K`",
    expect: "cost is <code>(1 - f**K) / (1 - f)</code> and the " +
            "<strong>success rate</strong> is <code>1 - f**K</code>",
  },
  {
    name: "two code spans each containing **",
    input: "`a**b` then `c**d`",
    expect: "<code>a**b</code> then <code>c**d</code>",
  },
  {
    name: "ordinary bold still renders",
    input: "this is **bold** text",
    expect: "this is <strong>bold</strong> text",
  },
  {
    name: "bold may still contain a code span",
    input: "**see `x` now**",
    expect: "<strong>see <code>x</code> now</strong>",
  },
  {
    name: "a lone ** inside a code span stays literal",
    input: "the value `0.2**3` is small",
    expect: "the value <code>0.2**3</code> is small",
  },
  {
    name: "html is escaped before anything else",
    input: "a <script>alert(1)</script> b",
    expect: "a &lt;script&gt;alert(1)&lt;/script&gt; b",
  },
  {
    name: "newlines become breaks",
    input: "one\ntwo",
    expect: "one<br>two",
  },
];

let failed = 0;
for (const c of cases) {
  const got = mdLite(c.input);
  if (got === c.expect) {
    console.log("  ok    " + c.name);
  } else {
    failed += 1;
    console.log("  FAIL  " + c.name);
    console.log("        input    " + JSON.stringify(c.input));
    console.log("        expected " + JSON.stringify(c.expect));
    console.log("        got      " + JSON.stringify(got));
  }
}

if (failed) {
  console.error("\n" + failed + " mdLite case(s) failed");
  process.exit(1);
}
console.log("\nall " + cases.length + " mdLite cases passed");
