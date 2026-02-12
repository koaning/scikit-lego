function render({ model, el }) {
  function draw() {
    el.innerHTML = "";
    const data = model.get("tree_data");
    if (!data || !data.nodes || data.nodes.length === 0) return;

    const DISPLAY_W = 700;
    const DISPLAY_H = 400;

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

    const panel = document.createElement("div");
    panel.classList.add("tree-info-panel");
    panel.textContent = "Click an edge to inspect it.";
    wrapper.appendChild(panel);

    // Build lookups
    const nodeMap = {};
    data.nodes.forEach((n) => {
      nodeMap[n.id] = n;
    });

    // Parent map for path computation
    const parentOf = {};
    data.edges.forEach((edge) => {
      parentOf[edge.target] = edge.source;
    });

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

    // Depth level lines
    const depthYs = {};
    data.nodes.forEach((n) => {
      if (depthYs[n.depth] === undefined) depthYs[n.depth] = n.y;
    });

    Object.entries(depthYs).forEach(([depth, y]) => {
      const d = parseInt(depth);
      if (d === 0) return; // skip root line
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", 0);
      line.setAttribute("y1", y);
      line.setAttribute("x2", data.width);
      line.setAttribute("y2", y);
      line.classList.add("depth-line");
      svg.appendChild(line);
    });

    // Depth level column labels (between each pair of consecutive depths)
    const sortedDepths = Object.keys(depthYs).map(Number).sort((a, b) => a - b);
    for (let i = 0; i < sortedDepths.length - 1; i++) {
      const d = sortedDepths[i];
      const midY = (depthYs[d] + depthYs[sortedDepths[i + 1]]) / 2;
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", 8);
      label.setAttribute("y", midY + 3);
      label.setAttribute("text-anchor", "start");
      label.classList.add("depth-label");
      label.textContent = `col ${d}`;
      svg.appendChild(label);
    }

    // Draw edges as clickable lines
    const edgeEls = {};
    const edgeData = {};
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
      const key = `${edge.source}-${edge.target}`;
      edgeEls[key] = line;
      edgeData[key] = edge;
    });

    // Draw nodes as small dots (not clickable)
    data.nodes.forEach((node) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", node.x);
      circle.setAttribute("cy", node.y);
      circle.setAttribute("r", 3);
      circle.classList.add("tree-node");
      if (node.is_leaf) circle.classList.add("tree-leaf");
      svg.appendChild(circle);
    });

    // SVG coordinate transform
    function toSVG(e) {
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
      return { x: svgPt.x, y: svgPt.y };
    }

    // Point-to-line-segment distance
    function distToSegment(px, py, x1, y1, x2, y2) {
      const dx = x2 - x1;
      const dy = y2 - y1;
      const lenSq = dx * dx + dy * dy;
      if (lenSq === 0) return Math.hypot(px - x1, py - y1);
      let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
    }

    // Find nearest edge
    function nearestEdge(px, py, maxDist) {
      let best = null;
      let bestKey = null;
      let bestDist = maxDist;
      Object.entries(edgeData).forEach(([key, edge]) => {
        const src = nodeMap[edge.source];
        const tgt = nodeMap[edge.target];
        const d = distToSegment(px, py, src.x, src.y, tgt.x, tgt.y);
        if (d < bestDist) {
          bestDist = d;
          best = edge;
          bestKey = key;
        }
      });
      return best;
    }

    function showEdgeInfo(edge, prefix) {
      if (!edge) {
        panel.textContent = "Click an edge to inspect it. Encoding: left = 0, right = 1.";
        return;
      }
      const parent = nodeMap[edge.source];
      const child = nodeMap[edge.target];
      const p = prefix ? prefix + " " : "";
      const dir = edge.side === "left" ? "left (0)" : "right (1)";
      // Build encoding path from root to this edge's target
      const path = pathToNode(edge.target);
      const encoding = [];
      for (let i = 0; i < path.length - 1; i++) {
        const key = `${path[i]}-${path[i + 1]}`;
        const e = edgeData[key];
        if (e) encoding.push(e.side === "left" ? "0" : "1");
      }
      panel.textContent = `${p}${parent.label} → ${dir} — ${child.samples} samples — col ${edge.depth} — path: [${encoding.join(", ")}]`;
    }

    // Highlight
    function applyHighlight(edge) {
      const targetId = edge ? edge.target : -1;
      const path = pathToNode(targetId);
      const pathSet = new Set(path);

      const pathEdgeKeys = new Set();
      for (let i = 0; i < path.length - 1; i++) {
        pathEdgeKeys.add(`${path[i]}-${path[i + 1]}`);
      }

      Object.entries(edgeEls).forEach(([key, line]) => {
        line.classList.toggle("highlighted", pathEdgeKeys.has(key));
      });
    }

    let selectedEdge = null;

    svg.addEventListener("mousemove", (e) => {
      const pt = toSVG(e);
      const edge = nearestEdge(pt.x, pt.y, 15);
      if (edge) {
        svg.style.cursor = "pointer";
        if (!selectedEdge) showEdgeInfo(edge, "");
      } else {
        svg.style.cursor = "";
        if (!selectedEdge) showEdgeInfo(null);
      }
    });

    svg.addEventListener("click", (e) => {
      const pt = toSVG(e);
      const edge = nearestEdge(pt.x, pt.y, 15);
      if (edge) {
        selectedEdge = edge;
        applyHighlight(edge);
        showEdgeInfo(edge, "Selected");
        model.set("selected_edge_target", edge.target);
        model.save_changes();
      } else {
        selectedEdge = null;
        applyHighlight(null);
        showEdgeInfo(null);
        model.set("selected_edge_target", -1);
        model.save_changes();
      }
    });

    model.on("change:selected_edge_target", () => {
      const targetId = model.get("selected_edge_target");
      if (targetId < 0) {
        selectedEdge = null;
        applyHighlight(null);
        showEdgeInfo(null);
      }
    });
  }

  draw();
  model.on("change:tree_data", draw);
}

export default { render };
