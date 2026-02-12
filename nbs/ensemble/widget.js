function render({ model, el }) {
  function draw() {
    el.innerHTML = "";
    const data = model.get("tree_data");
    if (!data || !data.nodes || data.nodes.length === 0) return;

    const DISPLAY_W = 700;
    const DISPLAY_H = 400;

    // Detect dark mode from marimo/document context
    const isDark =
      document.documentElement.dataset.colorMode === "dark" ||
      document.documentElement.classList.contains("dark") ||
      document.body.classList.contains("dark");

    const wrapper = document.createElement("div");
    wrapper.classList.add("tree-wrapper");
    if (isDark) wrapper.classList.add("dark");
    el.appendChild(wrapper);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", DISPLAY_W);
    svg.setAttribute("height", DISPLAY_H);
    svg.setAttribute("viewBox", `0 0 ${data.width} ${data.height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.classList.add("tree-widget");
    wrapper.appendChild(svg);

    // Info panel below SVG
    const panel = document.createElement("div");
    panel.classList.add("tree-info-panel");
    panel.textContent = "Click a node to inspect it.";
    wrapper.appendChild(panel);

    // Build lookups
    const nodeMap = {};
    data.nodes.forEach((n) => {
      nodeMap[n.id] = n;
    });

    // Build parent map from edges for path computation in JS
    const parentOf = {};
    data.edges.forEach((edge) => {
      parentOf[edge.target] = edge.source;
    });

    // Compute path from root to a node entirely in JS
    function pathToNode(nodeId) {
      if (nodeId < 0) return [];
      const path = [nodeId];
      let current = nodeId;
      while (parentOf[current] !== undefined) {
        current = parentOf[current];
        path.push(current);
      }
      path.reverse();
      return path;
    }

    // Draw edges
    const edgeEls = {};
    data.edges.forEach((edge) => {
      const src = nodeMap[edge.source];
      const tgt = nodeMap[edge.target];
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", src.x);
      line.setAttribute("y1", src.y);
      line.setAttribute("x2", tgt.x);
      line.setAttribute("y2", tgt.y);
      line.classList.add("tree-edge");
      svg.appendChild(line);
      edgeEls[`${edge.source}-${edge.target}`] = line;
    });

    // Draw nodes as small circles
    const nodeEls = {};
    const nodeR = 4;
    data.nodes.forEach((node) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", node.x);
      circle.setAttribute("cy", node.y);
      circle.setAttribute("r", nodeR);
      circle.classList.add("tree-node");
      if (node.is_leaf) circle.classList.add("tree-leaf");
      svg.appendChild(circle);
      nodeEls[node.id] = circle;
    });

    // Use SVG's native coordinate transform (guaranteed correct with viewBox)
    function toSVG(e) {
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
      return { x: svgPt.x, y: svgPt.y };
    }

    // Find nearest node to a point
    function nearestNode(px, py, maxDist) {
      let best = null;
      let bestDist = maxDist * maxDist;
      data.nodes.forEach((n) => {
        const dx = n.x - px;
        const dy = n.y - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) {
          bestDist = d2;
          best = n;
        }
      });
      return best;
    }

    // Show node info in panel
    function showInfo(node, prefix) {
      if (!node) {
        panel.textContent = "Click a node to inspect it.";
        return;
      }
      const p = prefix ? prefix + " " : "";
      if (node.is_leaf) {
        panel.textContent = `${p}Leaf — samples: ${node.samples}`;
      } else {
        panel.textContent = `${p}Split: ${node.label} — samples: ${node.samples}`;
      }
    }

    // Apply highlight for a given path + selected node (no Python needed)
    let currentPath = [];
    function applyHighlight(selectedId) {
      currentPath = pathToNode(selectedId);
      const pathSet = new Set(currentPath);

      const pathEdgeKeys = new Set();
      for (let i = 0; i < currentPath.length - 1; i++) {
        pathEdgeKeys.add(`${currentPath[i]}-${currentPath[i + 1]}`);
      }

      Object.entries(edgeEls).forEach(([key, line]) => {
        line.classList.toggle("highlighted", pathEdgeKeys.has(key));
      });

      Object.entries(nodeEls).forEach(([id, circle]) => {
        const nid = parseInt(id);
        circle.classList.toggle("on-path", pathSet.has(nid));
        circle.classList.toggle("selected", nid === selectedId);
      });

      if (selectedId >= 0 && nodeMap[selectedId]) {
        showInfo(nodeMap[selectedId], "Selected");
      } else {
        showInfo(null);
      }
    }

    // Hover: show info for nearest node
    svg.addEventListener("mousemove", (e) => {
      const pt = toSVG(e);
      const node = nearestNode(pt.x, pt.y, 20);
      if (node) {
        svg.style.cursor = "pointer";
        const sel = model.get("selected_node");
        if (sel < 0 || node.id !== sel) {
          showInfo(node, "");
        }
      } else {
        svg.style.cursor = "";
        const sel = model.get("selected_node");
        if (sel >= 0 && nodeMap[sel]) {
          showInfo(nodeMap[sel], "Selected");
        } else {
          showInfo(null);
        }
      }
    });

    // Click: select nearest node, highlight immediately in JS, then sync to Python
    svg.addEventListener("click", (e) => {
      const pt = toSVG(e);
      const node = nearestNode(pt.x, pt.y, 20);
      if (node) {
        // Highlight immediately — no Python roundtrip needed
        applyHighlight(node.id);
        model.set("selected_node", node.id);
        model.save_changes();
      } else {
        applyHighlight(-1);
        model.set("selected_node", -1);
        model.save_changes();
      }
    });

    // Also handle changes from Python side (e.g. programmatic selection)
    model.on("change:selected_node", () => {
      applyHighlight(model.get("selected_node"));
    });

    // Initial state
    applyHighlight(model.get("selected_node"));
  }

  draw();
  model.on("change:tree_data", draw);
}

export default { render };
