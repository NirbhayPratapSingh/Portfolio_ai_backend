// uploadData.js
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const { Pinecone } = require("@pinecone-database/pinecone");
const { GoogleGenerativeAI } = require("@google/generative-ai");
require("dotenv").config();

const app = express();
const upload = multer({ dest: "uploads/" });

// Init Pinecone
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.Index("portfolio-data");

// Init Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });

async function generateEmbedding(text) {
  const result = await embeddingModel.embedContent(text);
  return result.embedding.values;
}

app.post("/api/upload-pdf", upload.single("pdf"), async (req, res) => {
  try {
    const filePath = req.file.path;
    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdfParse(dataBuffer);
    const text = pdfData.text;

    // Optionally split text into chunks for embedding
    const chunkSize = 1000;
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
      chunks.push(text.substring(i, i + chunkSize));
    }

    const vectors = [];
    for (let i = 0; i < chunks.length; i++) {
      const embedding = await generateEmbedding(chunks[i]);
      vectors.push({
        id: `${req.file.filename}-${i}`,
        values: embedding,
        metadata: { text: chunks[i] },
      });
    }

    await index.upsert(vectors);
    fs.unlinkSync(filePath); // Clean up uploaded file
    res.json({ message: "PDF data uploaded to Pinecone", chunks: chunks.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = app;
