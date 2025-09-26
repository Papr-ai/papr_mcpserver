# MCP Server Testing Documentation

## Overview

The MCP server uses a comprehensive testing approach that merges multiple testing strategies into a single, powerful test suite.

## Test File: `test_mcp_comprehensive.py`

This single test file replaces three separate test files and provides complete validation of the MCP server.

### What It Merges

1. **`test_mcp_ci.py`** - CI/CD testing for automated pipelines
2. **`test_mcp_server.py`** - Server validation and command-line testing  
3. **MCP Protocol Testing** - End-to-end MCP client-server communication

### Test Coverage

The comprehensive test validates:

#### ✅ **Server Startup & Tool Registration**
- MCP server initialization
- All 8 tools properly registered
- Tool availability verification

#### ✅ **MCP Protocol Communication**
- Real MCP client-server communication
- Protocol compliance verification
- Tool discovery via MCP protocol

#### ✅ **API Connectivity**
- Papr Memory API integration
- Network connectivity testing
- API response validation

#### ✅ **Complete Workflow via MCP Protocol**
The test follows a realistic end-to-end workflow where each step uses results from previous steps:

1. **🧪 Step 1: CREATE (add_memory)**
   - Creates a new memory with test content
   - Extracts memory ID from the response
   - Validates successful creation

2. **🔍 Step 2: SEARCH (search_memory)**
   - Searches for the created memory
   - Extracts search_id from the response
   - Validates search functionality

3. **📖 Step 3: READ (get_memory)**
   - Retrieves the memory using ID from Step 1
   - Validates memory retrieval

4. **✏️ Step 4: UPDATE (update_memory)**
   - Updates the memory using ID from Step 1
   - Validates memory modification

5. **📝 Step 5: FEEDBACK (submit_feedback)**
   - Submits feedback using search_id from Step 2
   - Validates feedback functionality

6. **🗑️ Step 6: DELETE (delete_memory)**
   - Deletes the memory using ID from Step 1
   - Validates memory removal

**Workflow Pattern**: `add → search → get → update → feedback → delete`

### **Workflow Benefits**
- **Realistic Testing**: Mirrors actual usage patterns
- **Data Flow Validation**: Each step uses results from previous steps
- **ID Management**: Tests proper extraction and usage of memory IDs and search IDs
- **End-to-End Coverage**: Complete lifecycle from creation to deletion
- **Feedback Integration**: Tests feedback functionality with real search results

#### ✅ **Error Handling & Validation**
- Parameter validation (422 errors)
- 404 error handling
- Invalid input rejection
- Proper error propagation

#### ✅ **Server Command Validation**
- Command-line execution testing
- Import validation
- Server startup verification

#### ✅ **Tool Availability**
- All 8 tools callable
- Tool registration verification
- Function availability check

## Running the Test

```bash
# Run the comprehensive test
python test_mcp_comprehensive.py
```

## Test Results

The test provides detailed output showing:

- ✅ **7/7 tests passed** - Complete validation
- 📊 **Detailed logging** - Step-by-step progress
- 🔍 **Error analysis** - Specific failure points
- 📋 **Coverage summary** - What was tested

## Benefits of Merged Testing

### **Single Source of Truth**
- One test file instead of three
- Consistent testing approach
- Easier maintenance

### **Complete Coverage**
- Internal component testing
- MCP protocol communication
- Server command validation
- CRUD operations via MCP
- Error handling and validation
- API connectivity

### **CI/CD Ready**
- Automated testing support
- Exit codes for pipeline integration
- Comprehensive error reporting

### **Production Validation**
- Real-world usage scenarios
- End-to-end workflow testing
- MCP protocol compliance

## Test Architecture

```
test_mcp_comprehensive.py
├── Server Startup & Tool Registration
├── MCP Protocol Communication
├── API Connectivity Testing
├── Complete Workflow via MCP Protocol
│   ├── Step 1: CREATE (add_memory)
│   ├── Step 2: SEARCH (search_memory)
│   ├── Step 3: READ (get_memory)
│   ├── Step 4: UPDATE (update_memory)
│   ├── Step 5: FEEDBACK (submit_feedback)
│   └── Step 6: DELETE (delete_memory)
├── Error Handling & Validation
├── Server Command Validation
└── Tool Availability Check
```

## Exit Codes

- **0**: All tests passed - MCP server ready
- **1**: Some tests failed - MCP server issues

## Integration

The comprehensive test can be used in:

- **Development**: Quick validation during coding
- **CI/CD Pipelines**: Automated testing in GitHub Actions
- **Production**: Pre-deployment validation
- **Debugging**: Isolating specific functionality

## Dependencies

- `asyncio` - Async testing support
- `mcp` - MCP protocol client
- `papr_memory_mcp.core` - MCP server core
- `subprocess` - Command-line testing
- `json` - Response parsing
- `pathlib` - Path handling

## Environment Variables

The test uses environment variables from `.env`:
- `PAPR_API_KEY` - Papr Memory API key
- `MEMORY_SERVER_URL` - Papr Memory server URL

## Success Criteria

The test passes when:
- ✅ All 7 test categories pass
- ✅ MCP protocol communication works
- ✅ Complete workflow successful: add → search → get → update → feedback → delete
- ✅ Error handling working correctly
- ✅ API connectivity established
- ✅ All 8 tools available and callable
- ✅ Real memory IDs and search IDs properly extracted and used

This comprehensive approach ensures the MCP server is production-ready with complete functionality validation.
