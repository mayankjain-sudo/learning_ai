#!/usr/bin/env node

import { spawn } from 'child_process';

// Test cases to demonstrate different prompts
const testCases = [
  {
    name: 'greeting',
    args: { name: 'World' },
    description: 'Testing personalized greeting'
  },
  {
    name: 'explain-concept',
    args: { 
      concept: 'quantum computing', 
      audience: 'beginner' 
    },
    description: 'Testing concept explanation'
  }
];

let currentTest = 0;

async function runTest(testCase) {
  console.log(`\n=== ${testCase.description} ===`);
  console.log(`Prompt: ${testCase.name}`);
  console.log('Arguments:', JSON.stringify(testCase.args, null, 2));
  
  const serverProcess = spawn('node', ['server.js'], {
    cwd: process.cwd(),
    stdio: ['pipe', 'pipe', 'inherit']
  });

  return new Promise((resolve) => {
    let step = 0;

    serverProcess.stdout.on('data', (data) => {
      const response = data.toString().trim();
      step++;
      
      if (step === 1) {
        console.log('✓ Server initialized');
        // Send prompt request
        const promptRequest = {
          jsonrpc: '2.0',
          id: 2,
          method: 'prompts/get',
          params: {
            name: testCase.name,
            arguments: testCase.args
          }
        };
        
        console.log('→ Sending prompt to Ollama...');
        serverProcess.stdin.write(JSON.stringify(promptRequest) + '\n');
        
      } else if (step === 2) {
        console.log('✓ Got response from Ollama\n');
        try {
          const parsed = JSON.parse(response);
          if (parsed.result && parsed.result.messages) {
            console.log('RENDERED PROMPT:');
            console.log(parsed.result.messages[0].content.text);
            console.log('\nOLLAMA RESPONSE:');
            console.log(parsed.result.messages[1].content.text);
          }
        } catch (e) {
          console.log('Raw response:', response);
        }
        
        setTimeout(() => {
          serverProcess.kill();
          resolve();
        }, 100);
      }
    });

    // Initialize the server
    const initRequest = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'demo', version: '1.0.0' }
      }
    };

    serverProcess.stdin.write(JSON.stringify(initRequest) + '\n');

    // Safety timeout
    setTimeout(() => {
      console.log('Test timed out');
      serverProcess.kill();
      resolve();
    }, 20000);
  });
}

async function runAllTests() {
  console.log('=== MCP Ollama Demo ===');
  console.log('Testing multiple prompts with Ollama integration...\n');
  
  for (const testCase of testCases) {
    await runTest(testCase);
    
    // Brief pause between tests
    if (currentTest < testCases.length - 1) {
      console.log('\n' + '='.repeat(50));
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    currentTest++;
  }
  
  console.log('\n=== Demo completed! ===');
  process.exit(0);
}

runAllTests().catch(console.error);