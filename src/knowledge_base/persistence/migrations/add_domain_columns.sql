-- Add domain columns to support domain tagging
ALTER TABLE entities ADD COLUMN domain VARCHAR(100) NULL;
ALTER TABLE edges ADD COLUMN domain VARCHAR(100) NULL;
ALTER TABLE documents ADD COLUMN domain VARCHAR(100) NULL;

-- Create indexes for domain columns
CREATE INDEX idx_entities_domain ON entities(domain);
CREATE INDEX idx_edges_domain ON edges(domain);
CREATE INDEX idx_documents_domain ON documents(domain);