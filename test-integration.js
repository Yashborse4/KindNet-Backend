const fetch = require('node-fetch');

// Configuration
const BACKEND_URL = 'http://localhost:5000';
const TEST_TIMEOUT = 10000; // 10 seconds

// Test cases
const testCases = [
  {
    name: 'Safe message',
    text: 'Hello, how are you today?',
    expectedBullying: false
  },
  {
    name: 'Potentially harmful message',
    text: 'You are so stupid and ugly!',
    expectedBullying: true
  },
  {
    name: 'Moderate message',
    text: 'I disagree with your opinion',
    expectedBullying: false
  },
  {
    name: 'Strong harmful message',
    text: 'Kill yourself, loser!',
    expectedBullying: true
  },
  {
    name: 'Empty message',
    text: '',
    expectError: true
  }
];

// Color codes for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function healthCheck() {
  try {
    log('🔍 Checking backend health...', 'blue');
    const response = await fetch(`${BACKEND_URL}/`, {
      timeout: 5000
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    if (data.status === 'healthy') {
      log('✅ Backend is healthy and ready', 'green');
      return true;
    } else {
      log('❌ Backend returned unhealthy status', 'red');
      return false;
    }
  } catch (error) {
    log(`❌ Health check failed: ${error.message}`, 'red');
    return false;
  }
}

async function testDetection(testCase) {
  try {
    log(`\n📝 Testing: ${testCase.name}`, 'cyan');
    log(`   Text: "${testCase.text}"`, 'yellow');
    
    const startTime = Date.now();
    const response = await fetch(`${BACKEND_URL}/api/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: testCase.text,
        confidence_threshold: 0.7,
        include_details: true
      }),
      timeout: TEST_TIMEOUT
    });
    
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    if (testCase.expectError) {
      if (!response.ok) {
        log(`   ✅ Expected error received (${response.status})`, 'green');
        log(`   ⏱️  Response time: ${responseTime}ms`, 'blue');
        return { success: true, responseTime };
      } else {
        log(`   ❌ Expected error but got success`, 'red');
        return { success: false, responseTime };
      }
    }
    
    if (!response.ok) {
      log(`   ❌ HTTP Error: ${response.status} ${response.statusText}`, 'red');
      return { success: false, responseTime };
    }
    
    const data = await response.json();
    
    if (!data.success) {
      log(`   ❌ API Error: ${data.error}`, 'red');
      return { success: false, responseTime };
    }
    
    const result = data.data;
    const isBullying = result.is_bullying;
    const confidence = result.confidence;
    
    log(`   📊 Result: ${isBullying ? 'HARMFUL' : 'SAFE'} (${Math.round(confidence * 100)}% confidence)`, 
        isBullying ? 'red' : 'green');
    log(`   ⏱️  Response time: ${responseTime}ms`, 'blue');
    
    if (result.flagged_words && result.flagged_words.length > 0) {
      log(`   🚩 Flagged words: ${result.flagged_words.join(', ')}`, 'yellow');
    }
    
    if (result.openai_used) {
      log(`   🤖 OpenAI was used for analysis`, 'cyan');
    }
    
    // Check if result matches expectation
    const matches = isBullying === testCase.expectedBullying;
    if (matches) {
      log(`   ✅ Test PASSED - Result matches expectation`, 'green');
    } else {
      log(`   ❌ Test FAILED - Expected ${testCase.expectedBullying ? 'HARMFUL' : 'SAFE'}, got ${isBullying ? 'HARMFUL' : 'SAFE'}`, 'red');
    }
    
    return { 
      success: matches, 
      responseTime, 
      result: {
        isBullying,
        confidence,
        flaggedWords: result.flagged_words || [],
        openaiUsed: result.openai_used || false
      }
    };
    
  } catch (error) {
    log(`   ❌ Test Error: ${error.message}`, 'red');
    return { success: false, responseTime: 0, error: error.message };
  }
}

async function testBatchDetection() {
  try {
    log('\n🔄 Testing batch detection...', 'cyan');
    
    const texts = testCases.filter(tc => !tc.expectError).map(tc => tc.text);
    const startTime = Date.now();
    
    const response = await fetch(`${BACKEND_URL}/api/batch-detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        texts: texts,
        confidence_threshold: 0.7,
        include_details: true
      }),
      timeout: TEST_TIMEOUT * 2 // Allow more time for batch
    });
    
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    if (!response.ok) {
      log(`   ❌ HTTP Error: ${response.status} ${response.statusText}`, 'red');
      return { success: false, responseTime };
    }
    
    const data = await response.json();
    
    if (!data.success) {
      log(`   ❌ API Error: ${data.error}`, 'red');
      return { success: false, responseTime };
    }
    
    const batchResult = data.data;
    log(`   📊 Processed ${batchResult.total_processed} messages`, 'blue');
    log(`   🚨 ${batchResult.bullying_detected} messages flagged as harmful`, 'yellow');
    log(`   ⏱️  Total response time: ${responseTime}ms`, 'blue');
    log(`   ⚡ Average per message: ${Math.round(responseTime / texts.length)}ms`, 'blue');
    
    log(`   ✅ Batch detection test PASSED`, 'green');
    return { success: true, responseTime, batchResult };
    
  } catch (error) {
    log(`   ❌ Batch test error: ${error.message}`, 'red');
    return { success: false, responseTime: 0, error: error.message };
  }
}

async function testStats() {
  try {
    log('\n📈 Testing statistics endpoint...', 'cyan');
    
    const response = await fetch(`${BACKEND_URL}/api/stats`, {
      timeout: 5000
    });
    
    if (!response.ok) {
      log(`   ❌ HTTP Error: ${response.status} ${response.statusText}`, 'red');
      return { success: false };
    }
    
    const data = await response.json();
    
    if (!data.success) {
      log(`   ❌ API Error: ${data.error}`, 'red');
      return { success: false };
    }
    
    const stats = data.data;
    log(`   📊 Total requests: ${stats.total_requests || 0}`, 'blue');
    log(`   🚨 Bullying detected: ${stats.bullying_detected || 0}`, 'yellow');
    log(`   🎯 Average confidence: ${Math.round((stats.average_confidence || 0) * 100)}%`, 'blue');
    log(`   ✅ Statistics test PASSED`, 'green');
    
    return { success: true, stats };
    
  } catch (error) {
    log(`   ❌ Stats test error: ${error.message}`, 'red');
    return { success: false, error: error.message };
  }
}

async function runIntegrationTests() {
  log('🚀 Starting Cyberbullying Detection Integration Tests', 'bright');
  log('======================================================', 'bright');
  
  // Health check
  const isHealthy = await healthCheck();
  if (!isHealthy) {
    log('\n❌ Backend is not available. Please start the backend server first.', 'red');
    log('   Run: cd Backend && python app.py', 'yellow');
    process.exit(1);
  }
  
  // Wait a bit to ensure backend is fully ready
  await sleep(1000);
  
  const results = {
    passed: 0,
    failed: 0,
    totalTime: 0,
    details: []
  };
  
  // Test individual detection cases
  log('\n🧪 Running Individual Detection Tests', 'bright');
  log('=====================================', 'bright');
  
  for (const testCase of testCases) {
    const result = await testDetection(testCase);
    results.totalTime += result.responseTime;
    
    if (result.success) {
      results.passed++;
    } else {
      results.failed++;
    }
    
    results.details.push({
      testCase: testCase.name,
      success: result.success,
      responseTime: result.responseTime,
      error: result.error
    });
    
    // Small delay between tests
    await sleep(500);
  }
  
  // Test batch detection
  log('\n🔄 Running Batch Detection Test', 'bright');
  log('==============================', 'bright');
  
  const batchResult = await testBatchDetection();
  results.totalTime += batchResult.responseTime;
  
  if (batchResult.success) {
    results.passed++;
  } else {
    results.failed++;
  }
  
  // Test statistics
  log('\n📈 Running Statistics Test', 'bright');
  log('=========================', 'bright');
  
  const statsResult = await testStats();
  
  if (statsResult.success) {
    results.passed++;
  } else {
    results.failed++;
  }
  
  // Final summary
  log('\n📋 Test Summary', 'bright');
  log('==============', 'bright');
  log(`✅ Passed: ${results.passed}`, 'green');
  log(`❌ Failed: ${results.failed}`, 'red');
  log(`⏱️  Total time: ${results.totalTime}ms`, 'blue');
  log(`⚡ Average per test: ${Math.round(results.totalTime / testCases.length)}ms`, 'blue');
  
  if (results.failed === 0) {
    log('\n🎉 All tests passed! The system is working correctly.', 'green');
    log('✨ Your cyberbullying detection system is ready to use!', 'cyan');
  } else {
    log('\n⚠️  Some tests failed. Please check the backend logs and configuration.', 'yellow');
    
    // Show failed tests
    const failedTests = results.details.filter(d => !d.success);
    if (failedTests.length > 0) {
      log('\nFailed tests:', 'red');
      failedTests.forEach(test => {
        log(`  - ${test.testCase}: ${test.error || 'Unexpected result'}`, 'red');
      });
    }
  }
  
  log('\n📖 Next Steps:', 'bright');
  log('1. Start the frontend: cd Frontend && npm start', 'cyan');
  log('2. Open http://localhost:3000 in your browser', 'cyan');
  log('3. Test the chat interface with various messages', 'cyan');
  
  process.exit(results.failed === 0 ? 0 : 1);
}

// Run tests
runIntegrationTests().catch(error => {
  log(`\n💥 Unexpected error: ${error.message}`, 'red');
  process.exit(1);
});
