#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import fetch from 'node-fetch';

const OLLAMA_BASE_URL = 'http://localhost:11434';
const DEFAULT_MODEL = 'llama3.2';

async function sendToOllama(prompt, model = DEFAULT_MODEL) {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt, stream: false })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      return `HTTP Error ${response.status}: ${errorText}`;
    }
    
    const result = await response.json();
    return String(result.response || '');
  } catch (error) {
    return `Error: ${error.message}`;
  }
}

async function main() {
  const server = new McpServer({
    name: 'fixed-ollama-server',
    version: '1.0.0',
  });

  // Register prompts with handlers but no schemas
  server.registerPrompt(
    'greeting',
    {
      title: 'Personalized Greeting',
      description: 'Creates a personalized greeting'
    },
    async () => {
      const prompt = "Hello, Alice! Welcome to our MCP prompt demo.";
      const response = await sendToOllama(prompt);
      return { 
        messages: [
          { role: 'user', content: { type: 'text', text: prompt } },
          { role: 'assistant', content: { type: 'text', text: response } }
        ] 
      };
    }
  );

  server.registerPrompt(
    'explain-concept',
    {
      title: 'Concept Explainer',
      description: 'Explains complex concepts simply'
    },
    async () => {
      const prompt = "Please explain quantum computing in simple terms that a beginner could understand. Use examples and analogies where appropriate.";
      const response = await sendToOllama(prompt);
      return { 
        messages: [
          { role: 'user', content: { type: 'text', text: prompt } },
          { role: 'assistant', content: { type: 'text', text: response } }
        ] 
      };
    }
  );

  server.registerPrompt(
    'code-review',
    {
      title: 'Code Review Assistant',
      description: 'Reviews code and provides suggestions'
    },
    async () => {
      const prompt = `Please review the following javascript code and provide suggestions for improvement:

\`\`\`javascript
function hello() { console.log("hi") }
\`\`\`

Please focus on:
- Code quality and best practices
- Performance optimizations  
- Security considerations
- Readability and maintainability`;
      
      const response = await sendToOllama(prompt);
      return { 
        messages: [
          { role: 'user', content: { type: 'text', text: prompt } },
          { role: 'assistant', content: { type: 'text', text: response } }
        ] 
      };
    }
  );

  console.error('✅ All prompts registered');

  const transport = new StdioServerTransport();
  await server.connect(transport);
  
  console.error('✅ Server ready');
}

main().catch((error) => {
  console.error('❌ Server error:', error);
  process.exit(1);
});