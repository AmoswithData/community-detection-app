import React, { useEffect, useRef } from "react";
import Graph from "graphology";
import Sigma from "sigma";
import forceAtlas2 from "graphology-layout-forceatlas2";

const NetworkGraph = () => {
  const containerRef = useRef(null);

  useEffect(() => {
    // Create graph
    const graph = new Graph();

    // Example SNAP-like nodes
    graph.addNode("1", {
      label: "Node 1",
      size: 10
    });

    graph.addNode("2", {
      label: "Node 2",
      size: 10
    });

    graph.addNode("3", {
      label: "Node 3",
      size: 10
    });

    graph.addNode("4", {
      label: "Node 4",
      size: 10
    });

    // Add edges
    graph.addEdge("1", "2");
    graph.addEdge("1", "3");
    graph.addEdge("2", "4");
    graph.addEdge("3", "4");

    // Generate layout positions
    forceAtlas2.assign(graph, {
      iterations: 100
    });

    // Render graph
    const renderer = new Sigma(
      graph,
      containerRef.current
    );

    return () => {
      renderer.kill();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "700px",
        border: "1px solid #ddd"
      }}
    />
  );
};

export default NetworkGraph;