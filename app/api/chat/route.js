import { NextResponse } from "next/server";
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';
import fetch from "node-fetch";
import { GoogleGenerativeAI } from "@google/generative-ai";

const apiKey = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(apiKey);
const index = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  })
    .index("rag-index")
    .namespace("ns1");

const systemPrompt = `
# Rate My Professor Agent System Prompt

You are an AI assistant designed to help students find professors based on their specific queries. Your primary function is to use retrieval-augmented generation (RAG) to provide information about the top 3 most relevant professors for each user question.

## Your Capabilities:
1. Access a comprehensive database of professor information, including:
   - Name and title
   - Department and institution
   - Areas of expertise
   - Course ratings and reviews
   - Research interests and publications
   - Teaching style and methods

2. Understand and interpret various types of student queries, such as:
   - Specific subject areas or courses
   - Teaching styles or methods
   - Research opportunities
   - Difficulty level or workload
   - Personality traits or communication skills

3. Use RAG to retrieve and synthesize information from the database to provide accurate and relevant responses.

4. Present the top 3 most relevant professors for each query, along with a brief explanation of why they were selected.

## Your Responses Should:
1. Always provide exactly 3 professor recommendations, even if the match isn't perfect.
2. Include the following information for each recommended professor:
   - Name and title
   - Department and institution
   - A brief (2-3 sentence) explanation of why they match the query
   - An overall rating (out of 5 stars) based on student reviews
   - 1-2 key strengths or notable characteristics

3. Be concise yet informative, aiming for a total response length of 200-300 words.

4. Maintain a neutral and objective tone, avoiding biased language or personal opinions.

5. If the query is too vague or broad, ask for clarification before providing recommendations.

6. If asked about a specific professor not in your top 3 recommendations, provide information about that professor instead, following the same format as above.

7. Always respect privacy and avoid sharing any personal or sensitive information about professors or students.

## Example Interaction:
Human: I'm looking for a biology professor who specializes in marine ecosystems and has a reputation for being engaging in lectures.
`;

export async function POST(req) {
//     try {
//         const messages = await req.json();
//         const userMessage = messages[messages.length - 1];
//         const userQuery = userMessage.content;
    
//         const [queryEmbedding] = await fetchEmbeddingsWithRetry(userQuery);
//         const results = await index.query({
//           vector: queryEmbedding,
//           topK: 3,
//         });
    
//         const context = results.matches.map((match) => match.metadata.review).join("\n");
//         const prompt = `${systemPrompt}\n\n**Query:** ${userQuery}\n\n**Context:** ${context}`;
    
//         const response = await genAI.generateText({
//           prompt: prompt,
//           model: "text-davinci-003",
//           maxTokens: 500,
//         });
    

//       return NextResponse.json({ content: response.choices[0].text });
//     } catch (error) {
//       console.error("Error processing request:", error);
//       return NextResponse.json({ error: "Failed to process request." }, { status: 500 });
//     }
// }

    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      })
      const index = pc.index('rag').namespace('ns1')
      const openai = new OpenAI()
      
      const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
    encoding_format: 'float',
    })
    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
      })
      let resultString = ''
      results.matches.forEach((match) => {
        resultString += `
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
      })
      const lastMessage = data[data.length - 1]
const lastMessageContent = lastMessage.content + resultString
const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

const completion = await openai.chat.completions.create({
    messages: [
      {role: 'system', content: systemPrompt},
      ...lastDataWithoutLastMessage,
      {role: 'user', content: lastMessageContent},
    ],
    model: 'gpt-3.5-turbo',
    stream: true,
  })

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content
          if (content) {
            const text = encoder.encode(content)
            controller.enqueue(text)
          }
        }
      } catch (err) {
        controller.error(err)
      } finally {
        controller.close()
      }
    },
  })
  return new NextResponse(stream)

}