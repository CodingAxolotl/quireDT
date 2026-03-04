// quire.js

import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

const Quire = (function () {
  let embedder = null;
  let dataset = [];
  let datasetEmbeddings = [];
  let initialized = false;

  const DATASET_URL = "https://raw.githubusercontent.com/CodingAxolotl/QuireDT/refs/heads/main/dataset.json";

  async function init() {
    if (initialized) return;

    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

    const response = await fetch(DATASET_URL);
    dataset = await response.json();

    for (let item of dataset) {
      const result = await embedder(item.input, {
        pooling: "mean",
        normalize: true
      });
      datasetEmbeddings.push(result.data);
    }

    initialized = true;
    console.log("Quire initialized.");
  }

  function cosineSimilarity(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  async function ask(text) {
    if (!initialized) {
      throw new Error("Run await quire.init() first.");
    }

    const result = await embedder(text, {
      pooling: "mean",
      normalize: true
    });

    const inputEmbedding = result.data;

    let bestScore = -Infinity;
    for (let i = 0; i < datasetEmbeddings.length; i++) {
      const score = cosineSimilarity(inputEmbedding, datasetEmbeddings[i]);
      if (score > bestScore) bestScore = score;
    }

    const threshold = bestScore - 0.02;
    const matches = [];

    for (let i = 0; i < datasetEmbeddings.length; i++) {
      const score = cosineSimilarity(inputEmbedding, datasetEmbeddings[i]);
      if (score >= threshold) {
        matches.push(dataset[i].response);
      }
    }

    return matches[Math.floor(Math.random() * matches.length)];
  }

  return { init, ask };
})();

window.quire = Quire;
