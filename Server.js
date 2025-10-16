// server.js
const express = require("express");
const bodyParser  = require("body-parser");
require("dotenv").config();
const cors = require("cors");   // <-- add this

const app = express();

// ✅ Allow React frontend to talk to backend
app.use(cors({
  
  origin: ["http://localhost:5173","https://nirbhayportfolioai.netlify.app"],   // your frontend dev server
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"], // allow content-type
}));

// Import PDF upload route FIRST so multer can handle multipart requests
const uploadPdfApp = require("./uploadData");
app.use(uploadPdfApp);

app.use(bodyParser.json());

// Init Pinecone
const { Pinecone } = require("@pinecone-database/pinecone");
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.Index("portfolio-data");

// Init Gemini
const { GoogleGenerativeAI } = require("@google/generative-ai");
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
const chatModel = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

// Pre-prompt for Gemini
const prePrompt = `You are a highly experienced resume and career branding expert. Whenever you answer questions about Nirbhay, always highlight his strengths, achievements, and unique qualities in a compelling and professional manner. Your responses should be tailored to impress recruiters and hiring managers, making Nirbhay stand out as an exceptional candidate. Use persuasive language, focus on impact, and ensure every answer builds a strong, positive image of Nirbhay as a top talent.`;

// Create embeddings with Gemini
async function generateEmbedding(text) {
  const result = await embeddingModel.embedContent(text);
  return result.embedding.values;
}

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;

    // Step 1: Convert question → embedding
    const questionEmbedding = await generateEmbedding(message);

    // Step 2: Query Pinecone
    const results = await index.query({
      vector: questionEmbedding,
      topK: 5,
      includeMetadata: true,
    });

    // Step 3: Extract context
    const context = results.matches.map((match) => match.metadata.text).join("\n");

    // Step 4: Create prompt with prePrompt
    const prompt = `${prePrompt}\n\nBased on the following information about Nirbhay:\n${context}\n\nAnswer this question: ${message}`;

    // Step 5: Get response from Gemini
    const result = await chatModel.generateContent(prompt);
    const response = result.response.text();

    res.json({ response });
  } catch (err) {
    console.error("❌ Error:", err);
    res.status(500).json({ error: err.toString() });
  }
});

app.listen(3001, () => console.log("🚀 Server running on http://localhost:3001"));
