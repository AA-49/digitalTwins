(function () {
  "use strict";

  function ready(callback) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback, { once: true });
    } else {
      callback();
    }
  }

  function stringify(value, key) {
    if (value === null || value === undefined || value === "") return "Not available";
    if (typeof value === "number") {
      if (key.includes("probability") || key.includes("accuracy") || key.includes("recall") || key.includes("macro_f1") || key.includes("roc_auc")) {
        return `${(value * 100).toFixed(1)}%`;
      }
      return Number.isInteger(value) ? String(value) : value.toFixed(4);
    }
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function humanize(key) {
    return key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function appendDefinitionList(target, entries) {
    target.replaceChildren();
    entries.forEach(([key, value]) => {
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = humanize(key);
      detail.textContent = stringify(value, key);
      target.append(term, detail);
    });
    if (!entries.length) {
      const detail = document.createElement("dd");
      detail.textContent = "No additional details for this node.";
      target.append(detail);
    }
  }

  ready(function () {
    const graphElement = document.getElementById("knowledge-graph");
    const dataElement = document.getElementById("knowledge-graph-data");
    if (!graphElement || !dataElement) return;

    if (typeof window.cytoscape !== "function") {
      graphElement.textContent = "The interactive graph library could not be loaded. Use the accessible attribute list below.";
      graphElement.classList.add("kg-library-error");
      return;
    }

    let payload;
    try {
      payload = JSON.parse(dataElement.textContent);
    } catch (error) {
      graphElement.textContent = "The knowledge graph data could not be read.";
      return;
    }

    const nodeElements = payload.nodes.map((node) => {
      const classes = [node.type.toLowerCase()];
      if (node.type === "ShapContribution") classes.push(node.details.direction || "neutral");
      if (node.type === "RiskProbability" && node.details.selected) classes.push("selected-probability");
      return { data: node, classes: classes.join(" ") };
    });
    const edgeElements = payload.edges.map((edge) => ({ data: edge, classes: edge.group }));

    const cy = window.cytoscape({
      container: graphElement,
      elements: [...nodeElements, ...edgeElements],
      minZoom: 0.18,
      maxZoom: 2.5,
      wheelSensitivity: 0.22,
      boxSelectionEnabled: false,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#f8fbfd",
            "border-color": "#9fb2c3",
            "border-width": 1.5,
            "color": "#172133",
            "font-family": "system-ui, sans-serif",
            "font-size": 8,
            "label": "data(label)",
            "text-wrap": "wrap",
            "text-max-width": 90,
            "text-valign": "center",
            "text-halign": "center",
            "height": 38,
            "width": 100,
            "padding": 5,
          },
        },
        { selector: "node.patientsnapshot", style: { "background-color": "#172133", "border-color": "#172133", "color": "#ffffff", "font-size": 11, "font-weight": 800, "height": 65, "width": 135 } },
        { selector: "node.domain", style: { "background-color": "#dcebf5", "border-color": "#1769aa", "font-weight": 750, "height": 46, "width": 110 } },
        { selector: "node.observation", style: { "background-color": "#edf7fd", "border-color": "#1769aa" } },
        { selector: "node.attributedefinition", style: { "background-color": "#ffffff", "border-style": "dashed", "border-color": "#6f8292" } },
        { selector: "node.state", style: { "background-color": "#fff4de", "border-color": "#d8894b" } },
        { selector: "node.shapcontribution.supports", style: { "background-color": "#e0f3eb", "border-color": "#087f5b" } },
        { selector: "node.shapcontribution.opposes", style: { "background-color": "#fde7ea", "border-color": "#c92a3a" } },
        { selector: "node.shapcontribution.neutral", style: { "background-color": "#f1f3f5", "border-color": "#7b8794" } },
        { selector: "node.prediction", style: { "background-color": "#fff0cc", "border-color": "#c57a00", "font-size": 10, "font-weight": 800, "height": 56, "width": 130 } },
        { selector: "node.riskprobability", style: { "background-color": "#fffaf0", "border-color": "#d98b00" } },
        { selector: "node.riskprobability.selected-probability", style: { "border-width": 4, "font-weight": 800 } },
        { selector: "node.modelversion", style: { "background-color": "#ece7f7", "border-color": "#6f42c1" } },
        { selector: "node.modelevaluation", style: { "background-color": "#fff1db", "border-color": "#c57a00", "shape": "round-rectangle" } },
        { selector: "node.digitaltwin", style: { "background-color": "#efe8fb", "border-color": "#6f42c1", "border-width": 3, "font-weight": 800, "height": 58, "width": 130 } },
        {
          selector: "edge",
          style: {
            "curve-style": "bezier",
            "line-color": "#9fb2c3",
            "target-arrow-color": "#9fb2c3",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.65,
            "width": 1,
            "label": "data(label)",
            "font-size": 6,
            "color": "#5d6778",
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.82,
            "text-background-padding": 2,
            "text-rotation": "autorotate",
          },
        },
        { selector: "edge.profile", style: { "line-color": "#7fa9c8", "target-arrow-color": "#7fa9c8" } },
        { selector: "edge.supports", style: { "line-color": "#087f5b", "target-arrow-color": "#087f5b", "width": "mapData(weight, 0, 0.2, 1, 7)" } },
        { selector: "edge.opposes", style: { "line-color": "#c92a3a", "target-arrow-color": "#c92a3a", "width": "mapData(weight, 0, 0.2, 1, 7)" } },
        { selector: "edge.neutral", style: { "line-color": "#7b8794", "target-arrow-color": "#7b8794", "line-style": "dotted" } },
        { selector: "edge.prediction", style: { "line-color": "#d98b00", "target-arrow-color": "#d98b00" } },
        { selector: "edge.twin", style: { "line-color": "#6f42c1", "target-arrow-color": "#6f42c1", "width": 3 } },
        { selector: ".faded", style: { "opacity": 0.08, "text-opacity": 0.03 } },
        { selector: "node.focused", style: { "overlay-color": "#172133", "overlay-opacity": 0.12, "overlay-padding": 8, "border-width": 4 } },
      ],
      layout: {
        name: "cose",
        animate: false,
        randomize: false,
        fit: true,
        padding: 40,
        nodeRepulsion: 180000,
        idealEdgeLength: 105,
        edgeElasticity: 90,
        nestingFactor: 1.2,
        gravity: 0.15,
        numIter: 1300,
      },
    });

    const title = document.getElementById("kg-detail-title");
    const summary = document.getElementById("kg-detail-summary");
    const connections = document.getElementById("kg-detail-connections");
    const technical = document.getElementById("kg-detail-technical");
    const definition = document.getElementById("kg-detail-definition");
    const attributes = document.getElementById("kg-detail-attributes");
    const picker = document.getElementById("kg-node-picker");

    payload.nodes
      .slice()
      .sort((left, right) => left.label.localeCompare(right.label))
      .forEach((node) => {
        const option = document.createElement("option");
        option.value = node.id;
        option.textContent = `${node.type}: ${node.label}`;
        picker.append(option);
      });

    payload.attributes.forEach((attribute) => {
      const item = document.createElement("div");
      const heading = document.createElement("strong");
      const detail = document.createElement("span");
      heading.textContent = `${attribute.label}: ${attribute.value}`;
      detail.textContent = `${attribute.state}; SHAP ${attribute.shap_value >= 0 ? "+" : ""}${attribute.shap_value.toFixed(3)}`;
      item.append(heading, detail);
      attributes.append(item);
    });

    const definitionKeys = new Set(["feature", "raw_value", "decoded_state", "domain", "kind", "rule", "persistence", "feature_count"]);

    function selectNode(node) {
      cy.elements().removeClass("faded focused");
      const neighborhood = node.closedNeighborhood();
      cy.elements().difference(neighborhood).addClass("faded");
      node.addClass("focused");
      cy.animate({ fit: { eles: neighborhood, padding: 70 }, duration: 300 });

      const data = node.data();
      title.textContent = data.label;
      summary.textContent = data.summary || data.label;
      picker.value = data.id;

      connections.replaceChildren();
      const connectedEdges = node.connectedEdges();
      if (!connectedEdges.length) {
        const item = document.createElement("li");
        item.textContent = "No graph connections.";
        connections.append(item);
      } else {
        connectedEdges.forEach((edge) => {
          const item = document.createElement("li");
          const isSource = edge.source().id() === node.id();
          const other = isSource ? edge.target() : edge.source();
          item.textContent = `${isSource ? "→" : "←"} ${edge.data("label")} ${other.data("label")}`;
          connections.append(item);
        });
      }

      const detailEntries = Object.entries(data.details || {});
      appendDefinitionList(definition, detailEntries.filter(([key]) => definitionKeys.has(key)));
      appendDefinitionList(technical, detailEntries.filter(([key]) => !definitionKeys.has(key)));
    }

    cy.on("tap", "node", (event) => selectNode(event.target));
    cy.on("tap", (event) => {
      if (event.target === cy) {
        cy.elements().removeClass("faded focused");
        picker.value = "";
        cy.fit(undefined, 40);
      }
    });

    picker.addEventListener("change", () => {
      if (!picker.value) return;
      const node = cy.getElementById(picker.value);
      if (node.length) selectNode(node);
    });

    document.querySelectorAll("[data-kg-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        const filter = button.dataset.kgFilter;
        document.querySelectorAll("[data-kg-filter]").forEach((candidate) => {
          const active = candidate === button;
          candidate.classList.toggle("active", active);
          candidate.setAttribute("aria-pressed", String(active));
        });
        cy.nodes().forEach((node) => {
          const groups = node.data("groups") || [];
          const visible = filter === "all" || node.id() === "patient-current" || groups.includes(filter);
          node.style("display", visible ? "element" : "none");
        });
        cy.edges().forEach((edge) => {
          const visible = edge.source().style("display") !== "none" && edge.target().style("display") !== "none";
          edge.style("display", visible ? "element" : "none");
        });
        cy.elements().removeClass("faded focused");
        cy.layout({ name: "cose", animate: false, randomize: false, fit: true, padding: 40, nodeRepulsion: 180000, idealEdgeLength: 105, numIter: 900 }).run();
      });
    });

    document.querySelector("[data-kg-reset]").addEventListener("click", () => {
      cy.elements().removeClass("faded focused");
      picker.value = "";
      cy.fit(cy.elements(":visible"), 40);
    });

    const patient = cy.getElementById("patient-current");
    if (patient.length) selectNode(patient);
    window.patientKnowledgeGraph = cy;
  });
})();
