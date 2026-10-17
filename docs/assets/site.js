(function () {
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  if (btn) btn.addEventListener("click", function () {
    var current = root.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
  });
  // Collapse the module list by default on small screens so the article comes first.
  var box = document.querySelector(".sidebar-box");
  if (box && matchMedia("(max-width: 900px)").matches) box.removeAttribute("open");
})();
