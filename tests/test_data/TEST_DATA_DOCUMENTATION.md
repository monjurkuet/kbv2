# Test Data Directory - Phase 4 Ingestion Tests

## Directory Structure
`/home/muham/development/kbv2/test_data/`

## Available Test Documents

### Phase 4 Ingestion Test Files

1. **document_01_company_overview.txt** (2,300 bytes, 380 words)
   - Contains company overview information
   - Expected entities: TechCorp Solutions (ent_001), John Smith (ent_004), Sarah Johnson (ent_005)
   - Use case: Basic document ingestion, entity extraction testing

2. **document_02_quarterly_report.txt** (3,252 bytes, 520 words)
   - Contains quarterly financial data
   - Expected entities: TechCorp Solutions (ent_001), InnovateTech Inc (ent_002), Global Supply Co (ent_003)
   - Use case: Financial document processing, multi-entity extraction

3. **document_03_press_release.txt** (4,252 bytes, 680 words)
   - Contains press release content
   - Expected entities: TechCorp Solutions (ent_001), Global Supply Co (ent_003), John Smith (ent_004)
   - Use case: News/document processing, relationship mapping

4. **file_b_high_density.txt** (3,427 bytes, 550 words)
   - High-density text document with rich content
   - Expected entities: InnovateTech Inc (ent_002), Global Supply Co (ent_003), Sarah Johnson (ent_005)
   - Use case: Dense content processing, keyword extraction

5. **file_a_truly_low_density.txt** (12 bytes, ~2 words)
   - Minimal content document
   - Use case: Error handling, edge case testing for low-content documents

## Test Scenarios

### Basic Ingestion (All files)
- Verify file reading and text extraction
- Validate metadata extraction (size, word count, format)
- Confirm content hash generation

### Entity Extraction (Files 1-4)
- Test named entity recognition on varied content types
- Validate entity linking and disambiguation
- Verify relationship extraction between entities

### Edge Cases
- **file_a_truly_low_density.txt**: Test handling of minimal content
- Large file handling (monitor performance)
- Duplicate content detection via hash comparison

## Expected Document Count
- **Total files**: 5
- **Valid for ingestion**: 4 (documents 1-4 + file_b)
- **Edge case files**: 1 (file_a - should be processed but may have limited entities)

## Integration Points
- Frontend: Load via `frontend/tests/test-data/documents.json`
- Backend: Direct file system access to `/home/muham/development/kbv2/test_data/`
- Entity mapping: Cross-reference with `entities.json` for validation